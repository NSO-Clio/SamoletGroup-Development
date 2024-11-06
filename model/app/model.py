import pandas as pd
import numpy as np
from datetime import date, datetime
from catboost import CatBoostClassifier, Pool as CatBoostPool
import os

from pydantic import BaseModel
from pathlib import Path
from . import config
from .scoringModel import ScoringModel
from .modelGlobs import DATA_ALLOWED_COLUMNS

class ProcessingInputInvalid(Exception):
	pass

class Model:
	"""
	Processes the normalized data frame (after DataPreprocessor stage) and
	return the target output
	"""
	def __init__(self):
		self.model = ScoringModel(Path(config.get_settings().weights_path))
	
	def predict(self, Xdf: pd.DataFrame) -> np.ndarray:
		"""
		Makes prediction and returns the result.
		"""
		return self.model.predict_scoring(Xdf)
	
	def explain_rows(self, Xdf: pd.DataFrame) -> list[str]:
		"""
		Calculate the explanations of predictions.
		"""
		return self.model.explain_all(Xdf)

	def __call__(self, Xdf: pd.DataFrame) -> np.ndarray:
		return self.predict(Xdf)

class PredictionResult(BaseModel):
	contract_id: str
	report_date: str
	score: float
	
class Predictor():
	"""
	Pass user input data frame to this class and get output results.
	Basic class for model processing. Only this class should be used
	by API.
	"""
	
	def __init__(self):
		self.model = Model()

	def predict(self, input_df: pd.DataFrame) -> list[PredictionResult]:
		if not self.validate_input(input_df):
			raise ProcessingInputInvalid()
			
		outputs = self.model(input_df)

		res_df = pd.DataFrame([
			input_df['contract_id'], 
			input_df['report_date'],
			outputs
		]).transpose().rename(columns={'Unnamed 0': 'score'})

		results = res_df.to_dict(orient='index').values()
		results_pr = list(map(lambda x: PredictionResult(
			contract_id=str(x['contract_id']),
			report_date=str(x['report_date']),
			score=float(x['score'])
		), results))
		
		return results_pr
	
	def explain(self, input_df: pd.DataFrame) -> list[dict]:
		if not self.validate_input(input_df):
			raise ProcessingInputInvalid()

		exps = self.model.explain_rows(input_df)
		return exps

	def validate_input(self, input_df: pd.DataFrame) -> bool:
		if not np.array_equal(input_df.columns, DATA_ALLOWED_COLUMNS):
			return False

		return True
	
