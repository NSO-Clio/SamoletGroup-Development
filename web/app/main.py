from typing import Any
from flask.json import jsonify
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, send_file, url_for, flash
from flask_caching import Cache
import requests
from uuid import uuid4
import io
from pathlib import Path
import os
import json

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
        df = pd.read_csv(Path(Path(__file__).parent, "./static/default_submission.csv"))
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
        csv_data = pd.read_csv(file.stream)
        file.stream.seek(0)

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

        cache.set(f"raw_{ticket_id.__str__()}", csv_data)

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

def load_cached_ticket(ticket_id) -> Any:
    if not cache.has(ticket_id):
        if ticket_id == 'default':
            return load_default_submission()
        else:
            return None

    return cache.get(ticket_id)


@app.get('/tickets/<ticket_id>/')
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

@app.get('/tickets/<ticket_id>/raw')
def ticket_raw_file(ticket_id):
    data_df = load_cached_ticket(f"raw_{ticket_id}")
    if data_df is None:
        return redirect(url_for('index'))

    buf = io.BytesIO()
    data_df.to_csv(buf, index=False)
    buf.seek(0)

    return send_file(buf, download_name = f"raw_{ticket_id}.csv")

def calc_explanation(ticket_id, contract_id) -> dict | None:
    if ticket_id == "default":
        with open(Path(Path(__file__).parent, "./static/default_explanations.json"), 'r') as fd:
            data = json.load(fd)
            data = list(filter(lambda x: x['contract_id'] == int(contract_id), data))

    else:
        data_df: pd.DataFrame | None = load_cached_ticket(f"raw_{ticket_id}")
        if data_df is None:
            return None

        data_df = data_df[data_df['contract_id'] == int(contract_id)]
        if len(data_df) == 0:
            return None

        buf = io.BytesIO()
        data_df.to_csv(buf, index=False)
        buf.seek(0)

        resp = requests.post(
                f"{app.config['MODEL_HOST']}/explain",
                files={'file': buf}
        )

        resp.raise_for_status()

        data: list[dict] = resp.json()

    for dt in data:
        dt['columns_explanation'] = sorted(dt['columns_explanation'], key=lambda x: x['importance'], reverse=True)
        dcel = dt['columns_explanation']
        if len(dcel) > 10:
            dt['columns_explanation'] = dcel[:5] + dcel[-5:]

    return data

@app.get('/tickets/<ticket_id>/<contract_id>.json')
def ticket_contract_download(ticket_id, contract_id):
    exp = calc_explanation(ticket_id, contract_id)
    if exp == None:
        return redirect(url_for('ticket_view', ticket_id=ticket_id))

    return jsonify(exp)


@app.get('/tickets/<ticket_id>/<contract_id>')
def ticket_contract_explain(ticket_id, contract_id):
    exp = calc_explanation(ticket_id, contract_id)
    if exp == None:
        return redirect(url_for('ticket_view', ticket_id=ticket_id))

    return render_template('explanation.html', explanations=exp, ticket_id=ticket_id, contract_id=contract_id)
