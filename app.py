import os
import json
import time
import random
from time import monotonic
from typing import Dict, Any, List, Optional

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import requests

# =========================
# Bootstrap
# =========================
load_dotenv()
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# =========================
# Config / ENV
# =========================
# --- Hint providers ---
HINT_PROVIDER = os.getenv("HINT_PROVIDER", "").lower()  # 'openai' | 'gemini' | 'offline' | '' (auto)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # быстрый и бесплатный

# --- Judge0 (RapidAPI или self-hosted) ---
JUDGE0_URL = os.getenv("JUDGE0_URL", "https://judge0-ce.p.rapidapi.com")
JUDGE0_KEY = os.getenv("JUDGE0_KEY", "")
JUDGE0_HOST_HEADER = os.getenv("JUDGE0_HOST_HEADER", "judge0-ce.p.rapidapi.com")
JUDGE0_LANGUAGE_ID = int(os.getenv("JUDGE0_LANGUAGE_ID", "71"))  # 71 = Python 3.8+

DATA_DIR = os.getenv("DATA_DIR", "data")
USERS_JSON = os.path.join(DATA_DIR, "users.json")
TOPICS_JSON = os.path.join(DATA_DIR, "topics.json")

DEBUG_LOG = os.getenv("DEBUG_LOG", "0") == "1"

# =========================
# Ensure data exists
# =========================
DEFAULT_TOPICS = {
    "Python": {
        "Intro": {"tasks": ["task1"]},
        "Math": {"tasks": ["task2"]}
    }
}
DEFAULT_USERS = {
    "user1": {"completed_tasks": []}
}

os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(TOPICS_JSON):
    with open(TOPICS_JSON, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_TOPICS, f, ensure_ascii=False, indent=2)
if not os.path.exists(USERS_JSON):
    with open(USERS_JSON, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_USERS, f, ensure_ascii=False, indent=2)

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# =========================
# Tasks registry (desc + starter + tests)
# =========================
TASKS: Dict[str, Dict[str, str]] = {
    "task1": {
        "description": "Реализуй функцию my_len(s: str) -> int, которая возвращает длину строки без использования len().",
        "starter_code": """
# Напиши функцию ниже

def my_len(s: str) -> int:
    # TODO: посчитать длину s
    count = 0
    for _ in s:
        count += 1
    return count
""".strip(),
        "tests": """
# ---- ТЕСТЫ (не менять) ----
if __name__ == "__main__":
    assert my_len("") == 0
    assert my_len("abc") == 3
    assert my_len("привет") == 6
    print("OK")
""".strip()
    },
    "task2": {
        "description": "Реализуй функцию square(x: int|float) -> int|float, которая возвращает x*x.",
        "starter_code": """
# Напиши функцию ниже

def square(x):
    return x * x
""".strip(),
        "tests": """
# ---- ТЕСТЫ (не менять) ----
if __name__ == "__main__":
    assert square(2) == 4
    assert square(-3) == 9
    assert square(1.5) == 2.25
    print("OK")
""".strip()
    },
}

# =========================
# Simple rate limiter + hint cache
# =========================
RATE_LIMIT_WINDOW = 10.0  # сек
RATE_LIMIT_MAX = 3        # не более 3 вызовов за окно
_last_calls: Dict[str, List[float]] = {}

def allow_call(key: str) -> bool:
    now = monotonic()
    bucket = _last_calls.setdefault(key, [])
    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW:
        bucket.pop(0)
    if len(bucket) >= RATE_LIMIT_MAX:
        return False
    bucket.append(now)
    return True

HINT_CACHE_TTL = 60  # сек
_hint_cache: Dict[str, Dict[str, Any]] = {}  # key -> {"until": ts, "hint": str}

def get_cached_hint(key: str):
    item = _hint_cache.get(key)
    if not item:
        return None
    if time.time() > item["until"]:
        _hint_cache.pop(key, None)
        return None
    return item["hint"]

def set_cached_hint(key: str, hint: str):
    _hint_cache[key] = {"until": time.time() + HINT_CACHE_TTL, "hint": hint}

# =========================
# Hint providers
# =========================
class ProviderError(Exception):
    pass

def openai_chat_with_retry(messages, max_retries=5, timeout=30) -> dict:
    if not OPENAI_API_KEY:
        raise ProviderError("OPENAI_API_KEY is not set")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "messages": messages, "temperature": 0.2}

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt > max_retries:
                    # передадим тело выше (например insufficient_quota)
                    raise ProviderError(resp.text)
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = 0
                else:
                    base = min(8, 0.5 * (2 ** (attempt - 1)))
                    delay = base + random.uniform(0, 0.333 * base)
                if DEBUG_LOG:
                    print(f"[OpenAI] {resp.status_code}, retry in {delay:.2f}s ({attempt}/{max_retries})")
                time.sleep(delay)
                continue
            # другие статусы сразу превращаем в ошибку провайдера
            raise ProviderError(resp.text)
        except requests.RequestException as e:
            if attempt > max_retries:
                raise ProviderError(str(e))
            base = min(8, 0.5 * (2 ** (attempt - 1)))
            delay = base + random.uniform(0, 0.333 * base)
            if DEBUG_LOG:
                print(f"[OpenAI] network error {e}, retry in {delay:.2f}s ({attempt}/{max_retries})")
            time.sleep(delay)

def _norm_model(name: str) -> str:
    """Убираем префикс 'models/' если пришёл из ListModels."""
    name = (name or "").strip()
    return name[7:] if name.startswith("models/") else name

def gemini_generate(prompt: str, timeout=30) -> str:
    """
    Надёжный вызов Gemini v1: используем модель из .env и фоллбеки на семейство 2.x.
    ВАЖНО: URL — только /v1/, без v1beta.
    """
    if not GEMINI_API_KEY:
        raise ProviderError("GEMINI_API_KEY is not set")

    primary = _norm_model(os.getenv("GEMINI_MODEL") or "gemini-2.5-flash")

    # Фоллбеки по убыванию «универсальности»
    fallbacks = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-lite-001",
    ]
    # уникализируем
    seen = set(); models = []
    for m in [primary] + fallbacks:
        m = _norm_model(m)
        if m and m not in seen:
            seen.add(m); models.append(m)

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    last_err = None

    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={GEMINI_API_KEY}"
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                cands = data.get("candidates") or []
                if not cands:
                    raise ProviderError(f"No candidates ({model})")
                parts = cands[0].get("content", {}).get("parts") or []
                if not parts:
                    raise ProviderError(f"No parts ({model})")
                return (parts[0].get("text") or "").strip()
            last_err = f"{model}: {resp.status_code} {resp.text[:200]}"
            if os.getenv("DEBUG_LOG", "0") == "1":
                print("[Gemini] FAIL:", last_err)
        except requests.RequestException as e:
            last_err = f"{model}: {e}"
            if os.getenv("DEBUG_LOG", "0") == "1":
                print("[Gemini] NET ERR:", last_err)

    raise ProviderError(last_err or "Gemini not available")



@app.route("/diag/gemini/models")
def diag_gemini_models():
    if not GEMINI_API_KEY:
        return jsonify({"ok": False, "error": "GEMINI_API_KEY is not set"}), 500
    url = f"https://generativelanguage.googleapis.com/v1/models?key={GEMINI_API_KEY}"
    try:
        r = requests.get(url, timeout=20)
        return jsonify({"ok": r.status_code == 200, "status": r.status_code, "body": r.json()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def get_hint_text(description: str, code: str) -> str:
    """Определяет провайдера и выдаёт короткую подсказку."""
    system_msg = {
        "role": "system",
        "content": "Ты помогаешь студенту. Дай краткую подсказку (1–3 предложения), не раскрывая готового решения."
    }
    user_msg = {"role": "user", "content": f"Задание: {description}\n\nМой код:\n{code}"}

    provider_chain: List[str]
    if HINT_PROVIDER in ("openai", "gemini", "offline"):
        provider_chain = [HINT_PROVIDER]
    else:
        # авто: сначала OpenAI (если есть ключ), потом Gemini (если есть), потом offline
        provider_chain = []
        if OPENAI_API_KEY:
            provider_chain.append("openai")
        if GEMINI_API_KEY:
            provider_chain.append("gemini")
        provider_chain.append("offline")

    last_err: Optional[str] = None
    for prov in provider_chain:
        try:
            if prov == "openai":
                data = openai_chat_with_retry([system_msg, user_msg])
                return data["choices"][0]["message"]["content"].strip()
            elif prov == "gemini":
                prompt = f"Задание: {description}\n\nМой код:\n{code}\n\nДай подсказку кратко (1–3 предложения), без полного решения."
                return gemini_generate(prompt)
            else:
                # offline подсказка
                return "Подсказки офлайн: проверь, что возвращаемое значение соответствует тестам и граничным случаям."
        except ProviderError as e:
            last_err = str(e)
            if DEBUG_LOG:
                print(f"[Hint] provider {prov} error: {last_err}")
            continue
    # если все провалились
    raise RuntimeError(last_err or "No hint provider available")

# =========================
# Judge0 helpers
# =========================
def judge0_headers():
    headers = {"Content-Type": "application/json"}
    # RapidAPI mode
    if JUDGE0_KEY and JUDGE0_HOST_HEADER:
        headers.update({
            "x-rapidapi-key": JUDGE0_KEY,
            "x-rapidapi-host": JUDGE0_HOST_HEADER,
        })
    return headers

def run_in_judge0(source_code: str, stdin: str = "", expected_output: str = None) -> Dict[str, Any]:
    """Создаёт сабмишн и ждёт выполнения. При DEBUG_LOG печатает ответ."""
    create_url = f"{JUDGE0_URL}/submissions?base64_encoded=false&wait=true"
    payload = {"language_id": JUDGE0_LANGUAGE_ID, "source_code": source_code, "stdin": stdin}
    headers = judge0_headers()
    resp = requests.post(create_url, headers=headers, json=payload, timeout=60)
    if DEBUG_LOG:
        print("[J0] Status:", resp.status_code)
        print("[J0] Body:", resp.text[:500])
    resp.raise_for_status()
    return resp.json()

# =========================
# Routes
# =========================
@app.route("/")
def index_root():
    return render_template("index.html"), 200

@app.route("/api/knowledge-web")
def knowledge_web():
    topics = read_json(TOPICS_JSON)
    users = read_json(USERS_JSON)
    username = request.args.get("user", "user1")
    completed: List[str] = users.get(username, {}).get("completed_tasks", [])

    nodes = []
    edges = []
    topic_names = list(topics.keys())
    first_unlocked_given = False

    for tname in topic_names:
        tdata = topics[tname]
        topic_tasks = []
        for _, info in tdata.items():
            topic_tasks.extend(info.get("tasks", []))
        if topic_tasks and all(task in completed for task in topic_tasks):
            status = "completed"
        elif not first_unlocked_given:
            status = "unlocked"
            first_unlocked_given = True
        else:
            status = "locked"
        nodes.append({
            "data": {"id": tname, "label": tname, "status": status, "tasks": topic_tasks}
        })

    for i in range(len(topic_names) - 1):
        edges.append({"data": {"id": f"e{i}", "source": topic_names[i], "target": topic_names[i+1]}})

    return jsonify({"nodes": nodes, "edges": edges, "completed": completed})

@app.route("/api/task/<task_name>")
def get_task(task_name: str):
    task = TASKS.get(task_name)
    if not task:
        return jsonify({"error": "unknown_task"}), 404
    return jsonify({"task": task_name, "description": task["description"], "starter_code": task["starter_code"]})

@app.route("/run_code", methods=["POST"])
def run_code():
    data = request.get_json(force=True)
    code = data.get("code", "")
    task_name = data.get("task", "")
    task = TASKS.get(task_name)
    if not task:
        return jsonify({"message": "Unknown task"}), 400

    payload_code = f"{code}\n\n{task['tests']}\n"
    try:
        result = run_in_judge0(payload_code)
        stdout = (result.get("stdout") or "")
        stderr = (result.get("stderr") or "")
        compile_output = (result.get("compile_output") or "")
        status = result.get("status", {}).get("description")
        return jsonify({"stdout": stdout, "stderr": stderr, "compile_output": compile_output, "status": status})
    except requests.HTTPError as e:
        try:
            err_json = e.response.json()
        except Exception:
            err_json = None
        return jsonify({"message": "judge0_error", "details": err_json or str(e)}), 200
    except Exception as e:
        return jsonify({"message": "server_error", "details": str(e)}), 200

@app.route("/complete_task", methods=["POST"])
def complete_task():
    data = request.get_json(force=True)
    username = data.get("username", "user1")
    task = data.get("task")
    if not task:
        return jsonify({"error": "task_required"}), 400

    users = read_json(USERS_JSON)
    user = users.setdefault(username, {"completed_tasks": []})
    if task not in user["completed_tasks"]:
        user["completed_tasks"].append(task)
        write_json(USERS_JSON, users)
    return jsonify({"ok": True, "completed_tasks": user["completed_tasks"]})

@app.route("/api/get-hint", methods=["POST"])
def api_get_hint():
    # rate limit
    key = request.remote_addr or "anon"
    if not allow_call(key):
        return jsonify({"hint": None, "error": "rate_limited", "details": "Слишком часто. Попробуй через пару секунд."}), 200

    data = request.get_json(force=True)
    code = data.get("code", "")
    description = data.get("description", "")

    cache_key = f"{description}\n\n{code}"
    cached = get_cached_hint(cache_key)
    if cached is not None:
        return jsonify({"hint": cached, "cached": True})

    try:
        hint = get_hint_text(description, code)
        set_cached_hint(cache_key, hint)
        return jsonify({"hint": hint})
    except Exception as e:
        # не ломаем UI: возвращаем текст ошибки в details
        return jsonify({"hint": None, "error": "hint_error", "details": str(e)}), 200

# Доп: простая диагностика Judge0
@app.route("/diag/judge0")
def diag_judge0():
    try:
        res = run_in_judge0("print('OK')")
        return jsonify({
            "status": res.get("status", {}),
            "stdout": res.get("stdout"),
            "stderr": res.get("stderr"),
            "compile_output": res.get("compile_output"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask и так раздаёт /static, но оставим helper
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# =========================
# Entry
# =========================

@app.route("/ping")
def ping():
    # поможет убедиться, что именно ЭТОТ файл сейчас запущен
    return jsonify({
        "ok": True,
        "provider": (HINT_PROVIDER or "auto"),
        "cwd": os.getcwd()
    })

# УСТОЙЧИВЫЙ вызов Gemini — убедись, что функция gemini_generate определена выше!
@app.route("/diag/gemini")
def diag_gemini():
    try:
        txt = gemini_generate("Скажи 'OK' одним словом.")
        return jsonify({"ok": True, "hint": txt})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Показать все зарегистрированные маршруты текущего процесса
@app.route("/routes")
def routes():
    rules = []
    for r in app.url_map.iter_rules():
        rules.append({
            "rule": r.rule,
            "endpoint": r.endpoint,
            "methods": sorted([m for m in r.methods if m not in ("HEAD", "OPTIONS")])
        })
    # отсортируем по пути для удобства
    rules.sort(key=lambda x: x["rule"])
    return jsonify(rules)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    print("Working dir:", os.getcwd())
    app.run(host="0.0.0.0", port=port, debug=debug)
