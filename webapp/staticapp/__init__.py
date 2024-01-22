import os
from typing import Any
from flask import Flask
from config import Config

app: Any = Flask(__name__)
app.config.from_object(Config)


def get_realpath(path: str) -> str:
    base_dir: str = os.path.dirname(__file__)
    path: str = os.path.join(base_dir, path)
    return os.path.realpath(path)

from staticapp import views
