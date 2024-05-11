#!/usr/bin/env python

from enum import Enum
import os
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain.pydantic_v1 import BaseModel, Field
from langchain.agents import tool, AgentExecutor, create_openai_functions_agent
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
import pytz
from datetime import datetime, timedelta
from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockBarsRequest


# 1. Load Retriever
loader = WebBaseLoader(["https://www.investopedia.com/learn-how-to-trade-the-market-in-5-steps-4692230",
                        "https://www.investopedia.com/investing/complete-guide-choosing-online-stock-broker/",
                        "https://www.investopedia.com/best-online-brokers-4587872"])
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# 2. Crear Agentes
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information of trading. For any questions about trading, you must use this tool!",
)
search = TavilySearchResults()


class Intervalo(str, Enum):
    HOY = "hoy"
    TRES_DIAS = "3d"
    UNA_SEMANA = "1w"
    UN_MES = "1m"
    SEIS_MESES = "6m"
    UN_ANO = "1y"
    CINCO_ANOS = "5y"


data_client = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY_ID"], os.environ["ALPACA_API_SECRET_KEY"])


tz = pytz.timezone('America/New_York')


@tool
def get_stocks_from_MSFT_by_interval(intervalo: Intervalo) -> dict:
    """
    Returns historical stock data for Microsoft (MSFT) based on a specified time interval. It 
    can also be utilized for making predictions (use months or years for that). 

    Arguments:
        time: Time - An object containing information about the desired time interval.

    Returns:
        dict: A dictionary containing the stock bars data.
    """
    intervalo = intervalo

    timeframe = TimeFrame.Day
    if intervalo == Intervalo.HOY:
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        timeframe = TimeFrame.Hour
    elif intervalo == Intervalo.TRES_DIAS:
        start_time = datetime.now() - timedelta(days=3)
        timeframe = TimeFrame.Hour
    elif intervalo == Intervalo.UNA_SEMANA:
        start_time = datetime.now() - timedelta(weeks=1)
    elif intervalo == Intervalo.UN_MES:
        start_time = datetime.now() - timedelta(days=30)
    elif intervalo == Intervalo.SEIS_MESES:
        start_time = datetime.now() - timedelta(days=30*6)
        timeframe = TimeFrame.Week
    elif intervalo == Intervalo.UN_ANO:
        start_time = datetime.now() - timedelta(days=365)
        timeframe = TimeFrame.Week
    elif intervalo == Intervalo.CINCO_ANOS:
        start_time = datetime.now() - timedelta(days=365*5)
        timeframe = TimeFrame.Month

    start_time = pytz.timezone('America/New_York').localize(start_time)
    request_params = StockBarsRequest(
        symbol_or_symbols=['MSFT'],
        timeframe=timeframe,
        start=start_time
    )
    bars_df = data_client.get_stock_bars(
        request_params).df.tz_convert('America/New_York', level=1)

    bars_df.reset_index(inplace=True)

    bars_df['timestamp'] = bars_df['timestamp'].dt.strftime(
        '%Y-%m-%d %H:%M:%S')

    return bars_df.to_dict(orient='records')


tools = [retriever_tool, search, get_stocks_from_MSFT_by_interval]


# 3. Create Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"],
                 model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 4. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",  # Especifica el origen permitido
    allow_credentials=True,
    # Especifica los métodos permitidos
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],  # Permite todos los encabezados
)

# 5. Adding chain route

# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.


class Input(BaseModel):
    input: str
    chat_history: list[BaseMessage] = Field(
        ...,
    )

    def __init__(self, chat_history, **kwargs):
        super().__init__(chat_history=chat_history, **kwargs)
        converted_history = []
        for message in chat_history:
            if isinstance(message, dict):
                message_type = message.get("type")
                if message_type == "human":
                    converted_message = HumanMessage(
                        content=message["content"])
                elif message_type == "ai":
                    converted_message = AIMessage(content=message["content"])
                else:
                    raise ValueError(f"Unknown message type: {message_type}")
                converted_history.append(converted_message)
            else:
                converted_history.append(message)
        self.chat_history = converted_history


class Output(BaseModel):
    output: str


add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
