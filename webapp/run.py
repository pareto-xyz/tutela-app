import os
# from app import app      # full functioning app
from staticapp import app  # skeleton static app
from typing import Any


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    args: Any = parser.parse_args()
    port: int = int(os.environ.get('PORT', args.port))

    app.run(host='0.0.0.0', port=port)

