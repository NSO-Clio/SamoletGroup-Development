from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
	catboost_model_path: str = ""
	scoring_model_path: str = ""
	scoring_mean_median_imputer_path: str = ""

@lru_cache
def get_settings():
	return Settings()
