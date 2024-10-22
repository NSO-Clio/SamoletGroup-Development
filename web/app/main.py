import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, send_file, url_for, flash
from flask_caching import Cache
import requests
from uuid import uuid4
import io
from pathlib import Path
import os

import dotenv
dotenv.load_dotenv()
from . import config


app = Flask(__name__)
config.validate_config(app)
config.validate_cache_config(app)

cache = Cache(app)

# set default file
def load_default_submission():
    try:
        df = pd.read_csv(Path(Path(__file__).parent, "./static/submission.csv"))
        data = list(df.transpose().to_dict().values())
    except Exception as ex:
        print(f"Exception while loading default submission: {ex}")
        data = None

    cache.set("default", data)

    return data

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
        ticket_id = uuid4()
        cache.set(ticket_id.__str__(), data)

        return redirect(f"/tickets/{ticket_id.__str__()}")
    else:                
        flash('Невозможно обработать файл')
        return redirect(request.url)

@app.get('/upload')
def upload_view():
        return redirect(url_for('index'))

@app.get('/default')
def default_view():
    return redirect(url_for('ticket_view', ticket_id='default'))

def load_cached_ticket(ticket_id):
    if not cache.has(ticket_id):
        if ticket_id == 'default':
            return load_default_submission()
        else:
            return None

    return cache.get(ticket_id)


@app.get('/tickets/<ticket_id>')
def ticket_view(ticket_id):
    data = load_cached_ticket(ticket_id)
    if data is None:
        return redirect(url_for('index'))

    def normalize_data(x):
        s = x['report_date']
        x['report_date'] = s[8:10] + '.' + s[5:7] + '.' + s[0:4]
        x['score'] = round(x['score'] * 100)
        x['contract_id'] = int(x['contract_id'])
        

        return x

    data = list(map(normalize_data, data))

    return render_template('table.html', ticket_id=ticket_id, data=data)

@app.get('/tickets/<ticket_id>/csv')
def ticket_file(ticket_id):
    data = load_cached_ticket(ticket_id)
    if data is None:
        return redirect(url_for('index'))

    data_df = pd.DataFrame(data)
    buf = io.BytesIO()
    data_df.to_csv(buf, index=False)
    buf.seek(0)

    return send_file(buf, download_name = f"{ticket_id}.csv")



