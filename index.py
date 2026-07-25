"""
Vercel Python serverless entrypoint.

Vercel auto-detects a Flask `app` object in api/index.py and serves it as a
serverless function. Every request under /api/* is routed here (see vercel.json).
"""

import os
import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

from tasks import TASKS
from grader import grade_submission

app = Flask(__name__)
CORS(app)

SHEET_WEBHOOK_URL = os.environ.get("SHEET_WEBHOOK_URL", "").strip()


@app.route("/api/tasks")
def api_tasks():
    """Lets the frontend render task titles/prompts from a single source of truth."""
    return jsonify(TASKS)


@app.route("/api/submit", methods=["POST"])
def api_submit():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    answers = data.get("answers") or {}  # {"1": "...", "2": "...", ...}

    if not name or not email:
        return jsonify({"error": "Name and email are required."}), 400
    if "@" not in email:
        return jsonify({"error": "Please enter a valid email."}), 400

    try:
        result = grade_submission(name, email, answers)
    except Exception as exc:  # noqa: BLE001 - surface grading failures to the caller
        return jsonify({"error": f"Grading failed: {exc}"}), 500

    row_status = "not_configured"
    if SHEET_WEBHOOK_URL:
        row_status = _push_to_sheet(result)

    return jsonify({
        "results": result["results"],
        "total_score": result["total_score"],
        "max_score": result["max_score"],
        "percentage": result["percentage"],
        "overall_feedback": result["overall_feedback"],
        "sheet_status": row_status,
    })


def _push_to_sheet(result: dict) -> str:
    """POSTs one row to the Apps Script webhook. Returns 'ok' or an error string."""
    row = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "name": result["name"],
        "email": result["email"],
        "total_score": result["total_score"],
        "max_score": result["max_score"],
        "percentage": result["percentage"],
        "overall_feedback": result["overall_feedback"],
    }
    for task in TASKS:
        task_id = str(task["id"])
        task_result = result["results"].get(task_id, {"score": 0, "feedback": ""})
        row[f"task{task_id}_score"] = task_result["score"]
        row[f"task{task_id}_feedback"] = task_result["feedback"]

    try:
        resp = requests.post(SHEET_WEBHOOK_URL, json=row, timeout=15)
        if resp.status_code == 200:
            return "ok"
        return f"sheet_error_{resp.status_code}"
    except requests.RequestException as exc:
        return f"sheet_error_{exc}"
