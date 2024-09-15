import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "secret_key"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            # тут обращаемся к модельке
            df['report_date'] = df['report_date'].apply(lambda s: s[8:10] + '.' + s[5:7] + '.' + s[0:4])
            df['score'] = df['score'].apply(lambda x: round(x * 100))
            return render_template('table.html', data=df.to_dict(orient='records'))
        else:
            flash('Разрешены только CSV файлы')
            return redirect(request.url)
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)
