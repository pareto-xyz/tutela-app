from staticapp import app
from flask import render_template

PAGE_LIMIT = 50
HARD_MAX: int = 1000


@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

