from alpaca.data.requests import StockQuotesRequest, StockBarsRequest
from alpaca.data import StockHistoricalDataClient, TimeFrame
from enum import Enum
from datetime import datetime, timedelta
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pytz

tz = pytz.timezone('America/New_York')


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


ALPACA_API_KEY_ID = 'PK1GAMNH2PPG1NQLM98C'
ALPACA_API_SECRET_KEY = 'fZeBINKTPqeWU5GnTTYn3d3fDp9VYbDtIsHO2aXS'

data_client = StockHistoricalDataClient(
    ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY)


class Intervalo(str, Enum):
    HOY = "hoy"
    TRES_DIAS = "3d"
    UNA_SEMANA = "1w"
    UN_MES = "1m"
    SEIS_MESES = "6m"
    UN_ANO = "1y"
    CINCO_ANOS = "5y"


class Time(BaseModel):
    intervalo: Intervalo


@app.post("/")
async def root(time: Time):
    # Obtener el intervalo de tiempo deseado
    intervalo = time.intervalo

    timeframe = TimeFrame.Day
    # Calcular la fecha de inicio basada en el intervalo seleccionado
    if intervalo == Intervalo.HOY:
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        timeframe = TimeFrame.Minute
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
        symbol_or_symbols=['SPY'],
        timeframe=timeframe,
        start=start_time
    )
    bars_df = data_client.get_stock_bars(
        request_params).df.tz_convert('America/New_York', level=1)

    bars_df.reset_index(inplace=True)

    bars_data = bars_df.to_dict(orient='records')

    # Convertir a JSON y devolver
    return {"respuesta": bars_data}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001)
