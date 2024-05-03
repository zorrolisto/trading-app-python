from pydantic import BaseModel
from fastapi import FastAPI
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from fastapi.middleware.cors import CORSMiddleware

llm = Ollama(model="llama2")

print("loading web")
loader = WebBaseLoader(
    "https://www.investopedia.com/articles/trading/06/daytradingretail.asp")
print("web loaded")
docs = loader.load()

embeddings = OllamaEmbeddings()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
print("making vectors")
vector = FAISS.from_documents(documents, embeddings)
print("vectors done")

retriever = vector.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a bot, your answers are small and concise between 5 to 30 words ONLY, answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

global chat_history
chat_history = []

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Especifica el origen permitido
    allow_credentials=True,
    # Especifica los métodos permitidos
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],  # Permite todos los encabezados
)


class Message(BaseModel):
    messageForAI: str


@app.post("/")
async def root(message: Message):
    print("hello, I will think")
    res = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": message.messageForAI
    })
    respuesta = res["answer"]
    print("My answer is: ")
    print(respuesta)
    # chat_history.append(HumanMessage(content=message.messageForAI))
    # chat_history.append(AIMessage(content=respuesta))
    # print(len(chat_history))

    return {"respuesta": respuesta}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
