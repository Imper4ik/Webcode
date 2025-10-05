"""HTTP routes for the application."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
from flask import Blueprint, Flask, jsonify, render_template, request, send_from_directory

from .config import AppConfig
from .hints.providers import ProviderError, gemini_generate
from .hints.service import HintService
from .judge0 import Judge0Client
from .rate_limit import RateLimiter
from .storage import JsonStorage
from .tasks import TASKS


def register_routes(app: Flask, *, config: AppConfig, storage: JsonStorage, hint_service: HintService,
                    rate_limiter: RateLimiter, judge0: Judge0Client) -> None:
    bp = Blueprint("webcode", __name__)

    @bp.route("/")
    def index_root() -> tuple[str, int]:
        return render_template("index.html"), 200

    @bp.route("/api/knowledge-web")
    def knowledge_web() -> Any:
        topics = storage.read_topics()
        users = storage.read_users()
        username = request.args.get("user", "user1")
        completed: List[str] = users.get(username, {}).get("completed_tasks", [])

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        topic_names = list(topics.keys())
        first_unlocked_given = False

        for topic_name in topic_names:
            topic_data = topics[topic_name]
            topic_tasks: List[str] = []
            for info in topic_data.values():
                topic_tasks.extend(info.get("tasks", []))
            if topic_tasks and all(task in completed for task in topic_tasks):
                status = "completed"
            elif not first_unlocked_given:
                status = "unlocked"
                first_unlocked_given = True
            else:
                status = "locked"
            nodes.append({"data": {"id": topic_name, "label": topic_name, "status": status, "tasks": topic_tasks}})

        for idx in range(len(topic_names) - 1):
            edges.append({"data": {"id": f"e{idx}", "source": topic_names[idx], "target": topic_names[idx + 1]}})

        return jsonify({"nodes": nodes, "edges": edges, "completed": completed})

    @bp.route("/api/task/<task_name>")
    def get_task(task_name: str) -> Any:
        task = TASKS.get(task_name)
        if not task:
            return jsonify({"error": "unknown_task"}), 404
        return jsonify({"task": task_name, "description": task["description"], "starter_code": task["starter_code"]})

    @bp.route("/run_code", methods=["POST"])
    def run_code() -> Any:
        data = request.get_json(force=True)
        code = data.get("code", "")
        task_name = data.get("task", "")
        task = TASKS.get(task_name)
        if not task:
            return jsonify({"message": "Unknown task"}), 400

        payload_code = f"{code}\n\n{task['tests']}\n"
        try:
            result = judge0.run(payload_code)
            stdout = result.get("stdout") or ""
            stderr = result.get("stderr") or ""
            compile_output = result.get("compile_output") or ""
            status = result.get("status", {}).get("description")
            return jsonify({"stdout": stdout, "stderr": stderr, "compile_output": compile_output, "status": status})
        except requests.HTTPError as error:
            try:
                err_json = error.response.json()
            except Exception:
                err_json = None
            return jsonify({"message": "judge0_error", "details": err_json or str(error)}), 200
        except Exception as error:  # pragma: no cover - defensive
            return jsonify({"message": "server_error", "details": str(error)}), 200

    @bp.route("/complete_task", methods=["POST"])
    def complete_task() -> Any:
        data = request.get_json(force=True)
        username = data.get("username", "user1")
        task = data.get("task")
        if not task:
            return jsonify({"error": "task_required"}), 400

        users = storage.read_users()
        user = users.setdefault(username, {"completed_tasks": []})
        if task not in user["completed_tasks"]:
            user["completed_tasks"].append(task)
            storage.write_users(users)
        return jsonify({"ok": True, "completed_tasks": user["completed_tasks"]})

    @bp.route("/api/get-hint", methods=["POST"])
    def api_get_hint() -> Any:
        key = request.remote_addr or "anon"
        if not rate_limiter.allow(key):
            return jsonify({"hint": None, "error": "rate_limited", "details": "Слишком часто. Попробуй через пару секунд."}), 200

        data = request.get_json(force=True)
        code = data.get("code", "")
        description = data.get("description", "")

        try:
            hint, cached = hint_service.get_hint(description, code)
            return jsonify({"hint": hint, "cached": cached})
        except Exception as error:  # pragma: no cover - provider/network failure path
            return jsonify({"hint": None, "error": "hint_error", "details": str(error)}), 200

    @bp.route("/diag/judge0")
    def diag_judge0() -> Any:
        try:
            res = judge0.run("print('OK')")
            return jsonify({
                "status": res.get("status", {}),
                "stdout": res.get("stdout"),
                "stderr": res.get("stderr"),
                "compile_output": res.get("compile_output"),
            })
        except Exception as error:  # pragma: no cover - network failure path
            return jsonify({"error": str(error)}), 500

    @bp.route("/static/<path:filename>")
    def serve_static(filename: str):
        return send_from_directory(app.static_folder, filename)

    @bp.route("/ping")
    def ping() -> Any:
        return jsonify({"ok": True, "provider": config.hint.provider or "auto", "cwd": os.getcwd()})

    @bp.route("/diag/gemini/models")
    def diag_gemini_models() -> Any:
        if not config.hint.gemini_api_key:
            return jsonify({"ok": False, "error": "GEMINI_API_KEY is not set"}), 500
        url = f"https://generativelanguage.googleapis.com/v1/models?key={config.hint.gemini_api_key}"
        try:
            response = requests.get(url, timeout=20)
            body = response.json() if response.headers.get('Content-Type', '').startswith('application/json') else response.text
            return jsonify({"ok": response.status_code == 200, "status": response.status_code, "body": body})
        except Exception as error:  # pragma: no cover - network failure path
            return jsonify({"ok": False, "error": str(error)}), 500

    @bp.route("/diag/gemini")
    def diag_gemini() -> Any:
        if not config.hint.gemini_api_key:
            return jsonify({"ok": False, "error": "GEMINI_API_KEY is not set"}), 500
        try:
            text = gemini_generate("Скажи 'OK' одним словом.", config.hint, debug=config.debug_log)
            return jsonify({"ok": True, "hint": text})
        except ProviderError as error:  # pragma: no cover - network failure path
            return jsonify({"ok": False, "error": str(error)}), 500

    @bp.route("/routes")
    def routes() -> Any:
        rules: List[Dict[str, Any]] = []
        for rule in app.url_map.iter_rules():
            rules.append({
                "rule": rule.rule,
                "endpoint": rule.endpoint,
                "methods": sorted(m for m in rule.methods if m not in {"HEAD", "OPTIONS"}),
            })
        rules.sort(key=lambda item: item["rule"])
        return jsonify(rules)

    app.register_blueprint(bp)


__all__ = ["register_routes"]
