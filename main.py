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
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd


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

# 1er agente, experto trader
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information of trading. For any questions about trading, you must use this tool!",
)

# 2do agente, buscador de noticias actuales de trading
search = TavilySearchResults()


# 3cer agente comienza, AGENTE QUE USA ALPACA
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
    Returns historical stock data for Microsoft (MSFT) based on a specified time interval.
    This can't be used to predict, only to know information

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
# 3cer agente termina, AGENTE QUE USA ALPACA

# 4to agente para predecir


@tool
def get_prediction_for_tomorrow_in_MSFT_stock() -> dict:
    """
    HACE PREDICCION DEL DIA DE MÑN
    Retorna si es momento o no de comprar o vender, diciendo si las acciones bajaran o subiran mañana.
    Remember to inform the user that Alpaca API does not return data on Saturdays and Sundays; 
    it only provides data again after one month so predictions made are based on Friday if it's Saturday or Sunday."

    Add to the final of your response this: { prediction: prediction[0][0] }
    """

    fecha_hace_10_dias = datetime.now() - timedelta(days=10)
    fecha_hace_10_dias_str = fecha_hace_10_dias.strftime("%Y-%m-%d")

    request_params = StockBarsRequest(
        symbol_or_symbols=['MSFT'],
        timeframe=TimeFrame.Day,
        start=fecha_hace_10_dias_str
    )

    bars_df = data_client.get_stock_bars(
        request_params).df.tz_convert('America/New_York', level=1)
    df = pd.DataFrame(bars_df)
    df['return'] = np.log(df['close'] / df['close'].shift(1))

    lags, cols = 5, []
    for lag in range(1, lags + 1):
        col = f'lag_{lag}'
        df[col] = df['return'].shift(lag)
        cols.append(col)

    ultimo_lag_1 = df['lag_1'].iloc[-1]
    ultimo_lag_2 = df['lag_2'].iloc[-1]
    ultimo_lag_3 = df['lag_3'].iloc[-1]
    ultimo_lag_4 = df['lag_4'].iloc[-1]
    ultimo_lag_5 = df['lag_5'].iloc[-1]

    datos_pd = pd.DataFrame({
        'lag_1': [ultimo_lag_1],
        'lag_2': [ultimo_lag_2],
        'lag_3': [ultimo_lag_3],
        'lag_4': [ultimo_lag_4],
        'lag_5': [ultimo_lag_5]
    })

    modelo_cargado = load_model("modelo_entrenado.h5")
    prediction = modelo_cargado.predict(datos_pd)

    tendencia = "bajada" if prediction[0][0] < 0.5 else "subida"
    respuesta = "vender" if tendencia == "bajada" else "comprar"

    mensaje = f"Según el agente de DEEP LEARNING, la tendencia es de {tendencia} así que es momento de {respuesta} porque mañana {'bajará' if tendencia == 'bajada' else 'subirán'}, respuesta del bot {prediction[0][0]}"
    return mensaje


@tool
def calculate_earning_of_bought_dayA_and_sell_dayB(dayA: datetime, dayB: datetime) -> dict:
    """
        Return the costs for Day A and Day B and the profit that would 
        have been made by buying on Day A and selling on Day B
    """
    fechaA_mas_1_dia = dayA + timedelta(days=1)
    fechaA_mas_1_dia_str = fechaA_mas_1_dia.strftime("%Y-%m-%d")
    fechaB_mas_1_dia = dayB + timedelta(days=1)
    fechaB_mas_1_dia_str = fechaB_mas_1_dia.strftime("%Y-%m-%d")

    request_params = StockBarsRequest(
        symbol_or_symbols=['MSFT'],
        timeframe=TimeFrame.Day,
        start=dayA,
        end=fechaA_mas_1_dia_str
    )
    bars_df = data_client.get_stock_bars(
        request_params).df  # .tz_convert('America/New_York', level=1)
    bars_df.reset_index(inplace=True)
    recordsA = bars_df.to_dict(orient='records')

    request_params = StockBarsRequest(
        symbol_or_symbols=['MSFT'],
        timeframe=TimeFrame.Day,
        start=dayB,
        end=fechaB_mas_1_dia_str
    )
    bars_df = data_client.get_stock_bars(
        request_params).df  # .tz_convert('America/New_York', level=1)
    bars_df.reset_index(inplace=True)
    recordsB = bars_df.to_dict(orient='records')
    gananciaPorAccion = 0
    if len(recordsA) > 0 and len(recordsB) > 0:
        closeA = recordsA[0]["close"]
        closeB = recordsB[0]["close"]
        gananciaPorAccion = closeB - closeA
    return {
        'recordsA': recordsA,
        'recordsB': recordsB,
        'gananciaPorAccion': gananciaPorAccion,
    }


tools = [retriever_tool, search,
         calculate_earning_of_bought_dayA_and_sell_dayB,
         get_stocks_from_MSFT_by_interval,
         get_prediction_for_tomorrow_in_MSFT_stock]

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
    allow_origins=["*"],  # Especifica el origen permitido
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

    uvicorn.run(app, host="0.0.0.0", port=8000)
