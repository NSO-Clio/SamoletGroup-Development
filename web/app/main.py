import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
import requests

import dotenv
dotenv.load_dotenv()
from . import config

app = Flask(__name__)
config.validate_config(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.post('/upload')
def upload_file():
    file = request.files['file']
    if file:
        bff = file.stream

        resp = requests.post(
                f"{app.config['MODEL_HOST']}/predict",
                files={'file': file.stream}
        )

        if resp.status_code == 400:
            flash('Невозможно обработать файл')
            return redirect(request.url)

        resp.raise_for_status()

        data = resp.json()
        def normalize_data(x):
            s = x['report_date']
            x['report_date'] = s[8:10] + '.' + s[5:7] + '.' + s[0:4]
            x['score'] = round(x['score'] * 100)

            return x

        data = list(map(normalize_data, data))

        return render_template('table.html', data=data)
    else:                
        flash('Невозможно обработать файл')
        return redirect(request.url)

@app.get('/upload')
def upload_view():
        return redirect(url_for('index'))
