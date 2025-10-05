"""Flask entry point for the Webcode project."""
from __future__ import annotations

import os

from webcode import create_app

app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    print("Working dir:", os.getcwd())
    app.run(host="0.0.0.0", port=port, debug=debug)
