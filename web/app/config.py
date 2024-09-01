from flask import Flask

def validate_config(app: Flask):
    app.config.from_prefixed_env()
    app.config["MODEL_HOST"]
