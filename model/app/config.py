from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
	catboost_model_path: str = ""

@lru_cache
def get_settings():
	return Settings()
