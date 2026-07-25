"""
Grading pipeline built as a LangGraph graph.

Flow:
    START -> grade_all_tasks -> aggregate -> END

- grade_all_tasks: for every task with a non-empty submitted answer, calls the
  Cerebras-hosted gpt-oss-120b model (OpenAI-compatible endpoint) to score it 0-5
  against that task's "look for" rubric, and returns short feedback. The 17 calls
  run in parallel (Cerebras is fast, but this also keeps the whole request well
  inside a serverless function's execution time limit).
- aggregate: sums scores, computes a percentage, and writes a one-line overall verdict.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict, Dict, Any

from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from tasks import TASKS, MAX_TOTAL_SCORE

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
MODEL_NAME = "gpt-oss-120b"


def _client() -> OpenAI:
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is not set in the environment.")
    return OpenAI(api_key=api_key, base_url=CEREBRAS_BASE_URL)


class GradingState(TypedDict):
    name: str
    email: str
    answers: Dict[str, str]        # task_id (str) -> submitted text
    results: Dict[str, Any]        # task_id (str) -> {"score": int, "feedback": str}
    total_score: int
    max_score: int
    percentage: float
    overall_feedback: str


GRADER_SYSTEM_PROMPT = (
    "You are a strict but fair technical trainer grading how an employee used Claude / Claude Code "
    "for a specific hands-on task. You will be given the task instructions, the criteria the trainer "
    "is looking for, and the employee's pasted output (their transcript, notes, or result). "
    "Score ONLY what is shown — do not assume steps happened if they weren't described or pasted. "
    "Respond with STRICT JSON only, no markdown, in this exact shape: "
    '{"score": <integer 0-5>, "feedback": "<one or two sentence, specific, constructive feedback>"}'
)


def _grade_one(task: dict, submitted_text: str) -> dict:
    submitted_text = (submitted_text or "").strip()
    if not submitted_text:
        return {"score": 0, "feedback": "No answer submitted for this task."}

    user_prompt = (
        f"TASK: {task['title']}\n"
        f"WHAT THE EMPLOYEE WAS ASKED TO TYPE INTO CLAUDE: {task['prompt_given']}\n"
        f"WHAT TO LOOK FOR IN A GOOD SUBMISSION: {task['look_for']}\n\n"
        f"EMPLOYEE'S PASTED OUTPUT:\n\"\"\"\n{submitted_text}\n\"\"\"\n\n"
        "Score 0-5: 0 = blank/irrelevant, 1-2 = attempted but missing the core criteria, "
        "3 = partially meets criteria, 4 = meets criteria well, 5 = meets criteria fully with clear evidence."
    )

    client = _client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": GRADER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    raw = resp.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
        score = int(parsed.get("score", 0))
        score = max(0, min(5, score))
        feedback = str(parsed.get("feedback", "")).strip() or "No feedback returned."
    except (json.JSONDecodeError, ValueError, TypeError):
        score = 0
        feedback = f"Grader returned an unparseable response: {raw[:200]}"

    return {"score": score, "feedback": feedback}


def grade_all_tasks(state: GradingState) -> GradingState:
    results: Dict[str, Any] = {}
    tasks_to_grade = [t for t in TASKS if (state["answers"].get(str(t["id"]), "") or "").strip()]
    tasks_blank = [t for t in TASKS if t not in tasks_to_grade]

    for task in tasks_blank:
        results[str(task["id"])] = {"score": 0, "feedback": "No answer submitted for this task."}

    if tasks_to_grade:
        with ThreadPoolExecutor(max_workers=len(tasks_to_grade)) as pool:
            future_to_task = {
                pool.submit(_grade_one, task, state["answers"].get(str(task["id"]), "")): task
                for task in tasks_to_grade
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    results[str(task["id"])] = future.result()
                except Exception as exc:  # noqa: BLE001
                    results[str(task["id"])] = {"score": 0, "feedback": f"Grading error: {exc}"}

    state["results"] = results
    return state


def aggregate(state: GradingState) -> GradingState:
    total = sum(r["score"] for r in state["results"].values())
    state["total_score"] = total
    state["max_score"] = MAX_TOTAL_SCORE
    state["percentage"] = round(100 * total / MAX_TOTAL_SCORE, 1) if MAX_TOTAL_SCORE else 0.0

    if state["percentage"] >= 85:
        verdict = "Strong grasp of the daily Claude workflow habits."
    elif state["percentage"] >= 60:
        verdict = "Solid start — a few habits still need reinforcing."
    else:
        verdict = "Most habits not yet demonstrated — recommend redoing the sheet with real work."
    state["overall_feedback"] = verdict
    return state


def build_graph():
    graph = StateGraph(GradingState)
    graph.add_node("grade_all_tasks", grade_all_tasks)
    graph.add_node("aggregate", aggregate)
    graph.add_edge(START, "grade_all_tasks")
    graph.add_edge("grade_all_tasks", "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def grade_submission(name: str, email: str, answers: Dict[str, str]) -> GradingState:
    """Runs the LangGraph grading pipeline for one employee's submission."""
    graph = get_graph()
    initial_state: GradingState = {
        "name": name,
        "email": email,
        "answers": answers,
        "results": {},
        "total_score": 0,
        "max_score": MAX_TOTAL_SCORE,
        "percentage": 0.0,
        "overall_feedback": "",
    }
    return graph.invoke(initial_state)
