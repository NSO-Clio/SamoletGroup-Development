from flask import Flask

def validate_config(app: Flask):
    app.config.from_prefixed_env()
    app.config["MODEL_HOST"]

def validate_cache_config(app: Flask):
    if "REDIS_CONNECTION_URL" in app.config:
        print("Using Redis cache")
        cache_config = {
            "CACHE_TYPE": "RedisCache",
            "CACHE_REDIS_URL": app.config["REDIS_CONNECTION_URL"]
        }

        app.config.from_mapping(cache_config)
    else:
        print("Using in-memory cache")
        cache_config = {
            "CACHE_TYPE": "SimpleCache"
        }

        app.config.from_mapping(cache_config)
    
