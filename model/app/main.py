from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from fastapi import FastAPI, File, UploadFile, status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from uuid import UUID, uuid4
import asyncio
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import io
from typing import Literal

from .model import Predictor, ProcessingInputInvalid, PredictionResult
from . import config

app = FastAPI()
predictor = Predictor()

def read_csv(data: bytes) -> pd.DataFrame:
	bf = io.BytesIO(data)
	try:
		input_df = pd.read_csv(bf)
	except:
		raise HTTPException(status.HTTP_400_BAD_REQUEST)
	finally:
		bf.close()
	
	return input_df

def modelPredict(data: bytes) -> list[PredictionResult]:
	try:
		return predictor.predict(read_csv(data))
	except ProcessingInputInvalid:
		raise HTTPException(status.HTTP_400_BAD_REQUEST)

def modelExplain(data: bytes) -> list[dict[str, str]]:
	try:
		return predictor.explain(read_csv(data))
	except ProcessingInputInvalid:
		raise HTTPException(status.HTTP_400_BAD_REQUEST)

		
@app.post("/predict", responses={400: {"detail": "Bad Request", "description": "The payload file cannot be processed properly"}})
async def processTableFile(file: UploadFile) -> list[PredictionResult]:
	fileContents = await file.read()

	loop = asyncio.get_running_loop()
	with ProcessPoolExecutor() as pool:
		predicts = await loop.run_in_executor(
			pool, modelPredict, fileContents)


	return predicts

@app.post("/explain", responses={400: {"detail": "Bad Request", "description": "The payload file cannot be processed properly"}})
async def processTableFile(file: UploadFile) -> list[dict]:
	fileContents = await file.read()

	loop = asyncio.get_running_loop()
	with ProcessPoolExecutor() as pool:
		predicts = await loop.run_in_executor(
			pool, modelExplain, fileContents)


	return predicts
