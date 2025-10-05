"""Application factory for the Webcode project."""
from __future__ import annotations

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

from .config import AppConfig
from .hints.service import HintService
from .judge0 import Judge0Client
from .rate_limit import RateLimiter
from .routes import register_routes
from .storage import ensure_data


def create_app() -> Flask:
    load_dotenv()
    app = Flask(__name__, static_folder="static", template_folder="templates")
    CORS(app)

    config = AppConfig.from_env()
    storage = ensure_data(config)
    hint_service = HintService(config=config)
    rate_limiter = RateLimiter(max_calls=3, window_seconds=10.0)
    judge0 = Judge0Client(config=config)

    app.config['WEB_APP_CONFIG'] = config
    app.extensions.setdefault('webcode', {})
    app.extensions['webcode'].update({
        'config': config,
        'storage': storage,
        'hint_service': hint_service,
        'rate_limiter': rate_limiter,
        'judge0': judge0,
    })

    register_routes(
        app,
        config=config,
        storage=storage,
        hint_service=hint_service,
        rate_limiter=rate_limiter,
        judge0=judge0,
    )

    return app


__all__ = ["create_app", "AppConfig"]
