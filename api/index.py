"""
45-Minute Certification Exam backend.

- GET  /api/exam    -> sends questions WITHOUT correct answers (mcq) to the client
- POST /api/submit  -> scores MCQs server-side (deterministic, no API call, can't be
                        cheated by reading client JS), grades practicals via Cerebras
                        (gpt-oss-120b) through a small LangGraph graph, pushes one row
                        to the Google Sheet, and returns full per-question feedback so
                        the page can show the employee exactly what they got right/wrong.
"""

import os
import json
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict, Dict, Any, List

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# EXAM CONTENT
# ---------------------------------------------------------------------------

MCQS = [
    {
        "id": "m1", "type": "mcq", "source": "AI Dev Bible",
        "question": "Put the four-step agent loop in the order every serious AI coding tool follows:",
        "options": ["Write → Plan → Ship → Verify", "Plan → Write → Verify → Ship",
                    "Verify → Plan → Write → Ship", "Plan → Verify → Write → Ship"],
        "correct": 1, "points": 2,
    },
    {
        "id": "m2", "type": "mcq", "source": "AI Dev Bible",
        "question": "A model confidently stating a package name that doesn't actually exist is best described as:",
        "options": ["A prompt error", "A hallucination", "A context window overflow", "A token limit issue"],
        "correct": 1, "points": 2,
    },
    {
        "id": "m3", "type": "mcq", "source": "AI Dev Bible",
        "question": "What is CLAUDE.md for?",
        "options": ["A changelog of every past session",
                    "A file read automatically at the start of every session so stack/conventions don't need repeating",
                    "A list of banned commands", "The project's README for end users"],
        "correct": 1, "points": 2,
    },
    {
        "id": "m4", "type": "mcq", "source": "AI Dev Bible",
        "question": "Which of these is NOT one of the five ingredients of a strong prompt?",
        "options": ["Task", "Constraints", "Examples", "Deadline"],
        "correct": 3, "points": 2,
    },
    {
        "id": "m5", "type": "mcq", "source": "AI Dev Bible",
        "question": "The 5-line spec block is made of which fields?",
        "options": ["TASK / DATA / SERVICES / CONSTRAINTS / FORMAT", "GOAL / RISK / OWNER / DEADLINE / BUDGET",
                    "WHO / WHAT / WHEN / WHERE / WHY", "TASK / TEST / TIME / TEAM / TOOLS"],
        "correct": 0, "points": 2,
    },
    {
        "id": "m6", "type": "mcq", "source": "AI Dev Bible",
        "question": "Which command switches Claude Code into a mode where it proposes an approach but edits nothing until you approve?",
        "options": ["/compact", "/clear", "/plan (or Shift+Tab)", "/init"],
        "correct": 2, "points": 2,
    },
    {
        "id": "m7", "type": "mcq", "source": "AI Dev Bible",
        "question": "You're deep into the same ticket and the context bar is climbing. You then need to switch to a totally unrelated task. What's the right sequence?",
        "options": ["/clear now, then /compact later",
                    "/compact for the climbing context on the same ticket, /clear when you switch to the unrelated task",
                    "/init both times", "Neither — just keep going, context never needs managing"],
        "correct": 1, "points": 2,
    },
    {
        "id": "m8", "type": "mcq", "source": "AI Dev Bible",
        "question": "The \"kill repetition\" rule says a retyped instruction belongs in a saved command or Skill once you've typed a version of it:",
        "options": ["Once", "More than twice in a week", "Only if a teammate asks", "Every single session"],
        "correct": 1, "points": 2,
    },
    {
        "id": "m9", "type": "mcq", "source": "AI Dev Bible",
        "question": "In a Skill folder's progressive disclosure, what's loaded for every skill at session start, regardless of relevance?",
        "options": ["The full SKILL.md body", "All reference docs and scripts in the folder",
                    "Only the YAML frontmatter (name + description)", "Nothing until it's manually run"],
        "correct": 2, "points": 2,
    },
    {
        "id": "m10", "type": "mcq", "source": "AI Dev Bible",
        "question": "In the writer/reviewer pattern, what should you paste into the fresh review session?",
        "options": ["The whole conversation history from the writing session", "Only the diff — no backstory",
                    "A summary of what you think is wrong", "Nothing — just ask it to guess"],
        "correct": 1, "points": 2,
    },
    {
        "id": "m11", "type": "mcq", "source": "Claude Task Sheet",
        "question": "After connecting a real connector (GitHub or Google Drive), what question proves it's working with live data?",
        "options": ["\"What's the capital of France?\"", "\"What's currently open and waiting on my review?\"",
                    "\"Write me a poem\"", "\"What model are you?\""],
        "correct": 1, "points": 2,
    },
    {
        "id": "m12", "type": "mcq", "source": "Claude Task Sheet",
        "question": "Grounded strictly in \"Employees accrue 1 paid sick day per month during their first year, up to 12 days\" — how many sick days by month 3?",
        "options": ["1", "2", "3", "12"],
        "correct": 2, "points": 2,
    },
    {
        "id": "m13", "type": "mcq", "source": "Claude Task Sheet",
        "question": "You need a workflow that grades retrieved documents and, if none are relevant, rewrites the query and searches again. What's the deciding factor for reaching past a straight pipeline?",
        "options": ["The number of documents", "The need to loop back / branch conditionally",
                    "The size of the vector database", "Whether it's in Python or JavaScript"],
        "correct": 1, "points": 2,
    },
    {
        "id": "m14", "type": "mcq", "source": "Claude Task Sheet",
        "question": "\"Ask for proof, not a claim\" after a database-writing feature should include a request for:",
        "options": ["A confident one-line summary that it works",
                    "Actual test output, the row queried back for real, and any step that failed silently",
                    "A screenshot of the code", "Nothing — trust the agent's word"],
        "correct": 1, "points": 2,
    },
    {
        "id": "m15", "type": "mcq", "source": "Claude Task Sheet",
        "question": "What does running a second task in its own git worktree actually buy you?",
        "options": ["Faster internet",
                    "A second, fully isolated Claude Code session with zero risk of touching the main session's edits",
                    "Automatic code review", "A smaller context window"],
        "correct": 1, "points": 2,
    },
    {
        "id": "m16", "type": "mcq", "source": "MCP, Connectors & Plugins",
        "question": "Of the three doors into giving Claude a new tool, which one is \"one click, someone else already built and hosts it\"?",
        "options": ["MCP", "Plugin", "Connector", "Skill"],
        "correct": 2, "points": 2,
    },
    {
        "id": "m17", "type": "mcq", "source": "MCP, Connectors & Plugins",
        "question": "In a FastMCP server, what marks a plain Python function as a callable tool?",
        "options": ["A docstring", "The @mcp.tool() decorator", "Naming it main()", "Putting it in a file called tool.py"],
        "correct": 1, "points": 2,
    },
    {
        "id": "m18", "type": "mcq", "source": "MCP, Connectors & Plugins",
        "question": "You renamed a tool's parameter but the running Claude Code session still fails with a schema mismatch. Why?",
        "options": ["The tool schema is only fetched once, at connection time — you need to restart the session",
                    "MCP servers can never be edited after creation", "The tool name changed",
                    "Python doesn't allow renaming parameters"],
        "correct": 0, "points": 2,
    },
    {
        "id": "m19", "type": "mcq", "source": "MCP, Connectors & Plugins",
        "question": "In a PreToolUse hook script, which exit code actually blocks the tool call from running?",
        "options": ["exit 0", "exit 1", "exit 2", "Any nonzero code except 2"],
        "correct": 2, "points": 2,
    },
    {
        "id": "m20", "type": "mcq", "source": "MCP, Connectors & Plugins",
        "question": "Where does a PreToolUse hook get registered for a project?",
        "options": [".claude/settings.json", "CLAUDE.md", "package.json", "A global system environment variable"],
        "correct": 0, "points": 2,
    },
    {
        "id": "m21", "type": "mcq", "source": "RAG, LangChain & LangGraph",
        "question": "Chunks keep coming back mixing unrelated topics together, making retrieval imprecise. Are the chunks too big or too small, and what's the fix?",
        "options": ["Too small — merge chunks together", "Too big — use a smaller, more structure-aware chunk size",
                    "Just right — the vector database is broken", "Neither — this means the embedding model is broken"],
        "correct": 1, "points": 2,
    },
    {
        "id": "m22", "type": "mcq", "source": "RAG, LangChain & LangGraph",
        "question": "What does a text embedding actually represent?",
        "options": ["A compressed copy of the raw text",
                    "A numeric vector positioning the text's meaning in space, so similar meanings land near each other",
                    "A hash used only for deduplication", "The exact keywords in the text"],
        "correct": 1, "points": 2,
    },
    {
        "id": "m23", "type": "mcq", "source": "RAG, LangChain & LangGraph",
        "question": "Why do production vector databases use Approximate Nearest Neighbor (ANN) search instead of brute-force comparison?",
        "options": ["ANN is always 100% accurate, brute force isn't",
                    "Brute force doesn't scale past a few thousand vectors; ANN trades a little accuracy for large speed gains at scale",
                    "ANN is required by law for vector data", "Brute force only works with text, not vectors"],
        "correct": 1, "points": 2,
    },
    {
        "id": "m24", "type": "mcq", "source": "RAG, LangChain & LangGraph",
        "question": "You catch yourself writing if/else logic around a linear chain to decide what happens next. What's the signal telling you?",
        "options": ["Switch to a bigger LLM", "Reach for LangGraph instead of a plain chain",
                    "Add more vector database shards", "Increase the chunk size"],
        "correct": 1, "points": 2,
    },
    {
        "id": "m25", "type": "mcq", "source": "RAG, LangChain & LangGraph",
        "question": "You need the model to know about a policy that changes weekly. Do you reach for fine-tuning or RAG, and why?",
        "options": ["Fine-tuning — it's cheaper for fast-changing facts",
                    "RAG — facts update instantly by adding a document, no retraining needed",
                    "Neither works for changing facts", "Fine-tuning, because RAG can't cite sources"],
        "correct": 1, "points": 2,
    },
]

PRACTICALS = [
    {
        "id": "p1", "type": "practical", "source": "AI Dev Bible / Claude Task Sheet",
        "title": "Turn a Vague Ask Into a Tagged Prompt",
        "prompt_given": 'In Claude, rewrite this using task, context, constraints, and format: '
                         '"Can you add caching to the search endpoint, it\'s kind of slow, don\'t break the existing tests." '
                         "Paste Claude's actual response below.",
        "look_for": "A response clearly split into task / context / constraints / format sections, with nothing left "
                    "for Claude to guess at.",
        "max_score": 10,
    },
    {
        "id": "p2", "type": "practical", "source": "AI Dev Bible / Claude Task Sheet",
        "title": "Fill a 5-Line Spec Block",
        "prompt_given": "In Claude, write a TASK / DATA / SERVICES / CONSTRAINTS / FORMAT block for: when a user cancels "
                         "a subscription, write a row to a cancellations table and call the billing provider's refund API "
                         "for partial-month refunds. Paste the actual response below.",
        "look_for": "Five short labeled lines (TASK, DATA, SERVICES, CONSTRAINTS, FORMAT), each substantive and specific "
                    "to the cancellation/refund scenario.",
        "max_score": 10,
    },
    {
        "id": "p3", "type": "practical", "source": "MCP, Connectors & Plugins",
        "title": "Write and Run a Five-Line MCP Tool",
        "prompt_given": "Write a FastMCP server exposing a search_orders(status) tool, run python3 server.py to confirm "
                         "it starts without a traceback, then register it with claude mcp add and call it from a Claude "
                         "Code session. Paste the actual server.py code AND the transcript line showing the real tool call.",
        "look_for": "Working server.py code using @mcp.tool() and mcp.run(transport=\"stdio\"), plus evidence of a real "
                    "tool call appearing in the Claude Code transcript — not just the code with no proof it ran.",
        "max_score": 10,
    },
    {
        "id": "p4", "type": "practical", "source": "RAG, LangChain & LangGraph / Claude Task Sheet",
        "title": "Decide: One Path, or a Loop?",
        "prompt_given": "Ask Claude: I need a workflow that grades retrieved documents and, if none are relevant, "
                         "rewrites the query and searches again. Would a straight pipeline handle this, or do I need "
                         "something that can loop back a step — and what would you actually reach for? Paste the actual response.",
        "look_for": "A recommendation that explicitly names looping/conditional branching (e.g. LangGraph or an "
                    "equivalent graph-based orchestrator) as the deciding factor, not a plain linear pipeline.",
        "max_score": 10,
    },
    {
        "id": "p5", "type": "practical", "source": "AI Dev Bible",
        "title": "Ask for Proof, Not a Claim",
        "prompt_given": "On any real feature you're working on that involves a database write or external call, append: "
                         "\"When done, show me: 1) the actual test output, 2) the row it wrote, queried back for real, "
                         "3) any step that failed silently.\" Paste Claude's actual evidence-backed response below.",
        "look_for": "Real, checkable output was returned — actual test output and a real queried-back row — not just "
                    "an assertion that something works.",
        "max_score": 10,
    },
]

MCQ_TOTAL = sum(q["points"] for q in MCQS)
PRACTICAL_TOTAL = sum(p["max_score"] for p in PRACTICALS)
GRAND_TOTAL = MCQ_TOTAL + PRACTICAL_TOTAL
EXAM_DURATION_MINUTES = 45

# ---------------------------------------------------------------------------
# GRADER (practicals only — MCQs are scored deterministically below)
# ---------------------------------------------------------------------------

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
MODEL_NAME = "gpt-oss-120b"


def _client() -> OpenAI:
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is not set in the environment.")
    return OpenAI(api_key=api_key, base_url=CEREBRAS_BASE_URL)


GRADER_SYSTEM_PROMPT = (
    "You are a strict but fair technical trainer grading how an employee used Claude / Claude Code "
    "for a specific hands-on task. You will be given the task instructions, the criteria the trainer "
    "is looking for, and the employee's pasted output (their transcript, notes, or result). "
    "Score ONLY what is shown — do not assume steps happened if they weren't described or pasted. "
    "Respond with STRICT JSON only, no markdown, in this exact shape: "
    '{"score": <integer 0-10>, "feedback": "<one or two sentence, specific, constructive feedback>"}'
)


def _grade_practical(task: dict, submitted_text: str, max_retries: int = 4) -> dict:
    submitted_text = (submitted_text or "").strip()
    if not submitted_text:
        return {"score": 0, "feedback": "No answer submitted for this task."}

    user_prompt = (
        f"TASK: {task['title']}\n"
        f"WHAT THE EMPLOYEE WAS ASKED TO DO: {task['prompt_given']}\n"
        f"WHAT TO LOOK FOR IN A GOOD SUBMISSION: {task['look_for']}\n\n"
        f"EMPLOYEE'S PASTED OUTPUT:\n\"\"\"\n{submitted_text}\n\"\"\"\n\n"
        "Score 0-10: 0 = blank/irrelevant, 1-4 = attempted but missing the core criteria, "
        "5-7 = partially meets criteria, 8-9 = meets criteria well, 10 = meets criteria fully with clear evidence."
    )

    client = _client()
    raw = None
    last_error = None
    for attempt in range(max_retries):
        try:
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
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1) if is_rate_limit else 1)
                continue
            return {"score": 0, "feedback": f"Grader unavailable after retries: {exc}"}

    if raw is None:
        return {"score": 0, "feedback": f"Grader unavailable after retries: {last_error}"}

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
        score = int(parsed.get("score", 0))
        score = max(0, min(10, score))
        feedback = str(parsed.get("feedback", "")).strip() or "No feedback returned."
    except (json.JSONDecodeError, ValueError, TypeError):
        score = 0
        feedback = f"Grader returned an unparseable response: {raw[:200]}"

    return {"score": score, "feedback": feedback}


class ExamState(TypedDict):
    name: str
    email: str
    mcq_answers: Dict[str, int]
    practical_answers: Dict[str, str]
    mcq_results: Dict[str, Any]
    practical_results: Dict[str, Any]
    mcq_score: int
    practical_score: int
    total_score: int
    percentage: float
    overall_feedback: str


def score_mcqs(state: ExamState) -> ExamState:
    results = {}
    total = 0
    for q in MCQS:
        submitted = state["mcq_answers"].get(q["id"])
        is_correct = submitted is not None and int(submitted) == q["correct"]
        points = q["points"] if is_correct else 0
        total += points
        results[q["id"]] = {
            "correct": is_correct,
            "points": points,
            "max_points": q["points"],
            "correct_answer": q["options"][q["correct"]],
        }
    state["mcq_results"] = results
    state["mcq_score"] = total
    return state


def grade_practicals(state: ExamState) -> ExamState:
    results: Dict[str, Any] = {}
    tasks_to_grade = [t for t in PRACTICALS if (state["practical_answers"].get(t["id"], "") or "").strip()]
    tasks_blank = [t for t in PRACTICALS if t not in tasks_to_grade]

    for task in tasks_blank:
        results[task["id"]] = {"score": 0, "feedback": "No answer submitted for this task."}

    if tasks_to_grade:
        max_workers = min(3, len(tasks_to_grade))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_task = {}
            for task in tasks_to_grade:
                future_to_task[pool.submit(_grade_practical, task, state["practical_answers"].get(task["id"], ""))] = task
                time.sleep(0.3)
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    results[task["id"]] = future.result()
                except Exception as exc:  # noqa: BLE001
                    results[task["id"]] = {"score": 0, "feedback": f"Grading error: {exc}"}

    state["practical_results"] = results
    state["practical_score"] = sum(r["score"] for r in results.values())
    return state


def aggregate(state: ExamState) -> ExamState:
    total = state["mcq_score"] + state["practical_score"]
    state["total_score"] = total
    state["percentage"] = round(100 * total / GRAND_TOTAL, 1) if GRAND_TOTAL else 0.0

    if state["percentage"] >= 85:
        verdict = "Certified — strong grasp across prompting, MCP, and RAG orchestration."
    elif state["percentage"] >= 70:
        verdict = "Passed — solid grasp, a few areas worth revisiting."
    elif state["percentage"] >= 50:
        verdict = "Borderline — review the missed sections and retake."
    else:
        verdict = "Not yet passing — recommend re-reading the source material and retaking."
    state["overall_feedback"] = verdict
    return state


_GRAPH = None


def build_graph():
    graph = StateGraph(ExamState)
    graph.add_node("score_mcqs", score_mcqs)
    graph.add_node("grade_practicals", grade_practicals)
    graph.add_node("aggregate", aggregate)
    graph.add_edge(START, "score_mcqs")
    graph.add_edge("score_mcqs", "grade_practicals")
    graph.add_edge("grade_practicals", "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile()


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def grade_exam(name: str, email: str, mcq_answers: Dict[str, int], practical_answers: Dict[str, str]) -> ExamState:
    graph = get_graph()
    initial_state: ExamState = {
        "name": name, "email": email,
        "mcq_answers": mcq_answers, "practical_answers": practical_answers,
        "mcq_results": {}, "practical_results": {},
        "mcq_score": 0, "practical_score": 0, "total_score": 0,
        "percentage": 0.0, "overall_feedback": "",
    }
    return graph.invoke(initial_state)


# ---------------------------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------------------------

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Claude & Claude Code — Certification Exam</title>
<style>
  :root{
    --ink:#1b1f23; --paper:#eef0ec; --card:#ffffff; --rule:#d7dcd3;
    --accent:#2f5d50; --accent-soft:#e4ede9; --warn:#a6432f; --good:#2f5d50;
    --mono:"IBM Plex Mono","SF Mono",Menlo,Consolas,monospace;
    --serif:"IBM Plex Serif",Georgia,serif;
    --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  }
  *{box-sizing:border-box;}
  body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);line-height:1.5;}
  header{padding:40px 24px 20px;max-width:880px;margin:0 auto;}
  .eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 10px;}
  h1{font-family:var(--serif);font-size:clamp(26px,4vw,38px);margin:0 0 10px;letter-spacing:-0.01em;}
  header p{max-width:64ch;color:#4a504a;margin:0;}

  #timerBar{
    position:sticky; top:0; z-index:50;
    background:var(--card); border-bottom:1px solid var(--rule);
    padding:10px 24px; display:flex; justify-content:space-between; align-items:center;
    font-family:var(--mono); font-size:13px;
  }
  #timerBar.warn{ background:#fbeae6; }
  #timeLeft{ font-weight:700; font-size:16px; }
  #progressText{ color:#6b6f68; }

  main{max-width:880px;margin:0 auto;padding:20px 24px 120px;}
  .section-header{
    font-family:var(--mono); font-size:12px; letter-spacing:.12em; text-transform:uppercase;
    color:var(--accent); margin:36px 0 6px; padding-bottom:8px; border-bottom:1px solid var(--rule);
  }
  .section-sub{ color:#6b6f68; font-size:13px; margin:0 0 16px; }

  .identity{
    background:var(--card); border:1px solid var(--rule); border-radius:4px; padding:24px;
    margin-bottom:20px; display:grid; grid-template-columns:1fr 1fr; gap:16px;
  }
  @media (max-width:560px){ .identity{grid-template-columns:1fr;} }
  .field label{display:block;font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#6b6f68;margin-bottom:6px;}
  .field input, .field textarea{
    width:100%; font-family:var(--sans); font-size:15px; padding:10px 12px;
    border:1px solid var(--rule); border-radius:3px; background:#fbfbf9; color:var(--ink);
  }
  .field input:focus, .field textarea:focus{ outline:2px solid var(--accent); outline-offset:1px; background:#fff; }

  .q-card, .task-card{
    background:var(--card); border:1px solid var(--rule); border-left:3px solid var(--accent);
    border-radius:4px; padding:18px 20px; margin-bottom:14px;
  }
  .q-meta{ font-family:var(--mono); font-size:11px; color:var(--accent); text-transform:uppercase; letter-spacing:.06em; }
  .q-text{ font-family:var(--serif); font-size:17px; margin:6px 0 12px; }
  .options label{
    display:flex; gap:10px; align-items:flex-start; padding:8px 10px; border-radius:3px; cursor:pointer; font-size:14.5px;
  }
  .options label:hover{ background:var(--accent-soft); }
  .options input{ margin-top:3px; }

  .task-title{ font-family:var(--serif); font-size:18px; margin:2px 0 10px; }
  .task-prompt{
    font-family:var(--mono); font-size:12.5px; color:#464b45; background:var(--accent-soft);
    border-radius:3px; padding:10px 12px; margin:0 0 12px; white-space:pre-wrap;
  }
  .task-card textarea{
    width:100%; min-height:110px; resize:vertical; font-family:var(--mono); font-size:13.5px;
    padding:12px; border:1px solid var(--rule); border-radius:3px; background:#fbfbf9;
  }
  .task-card textarea:focus{ outline:2px solid var(--accent); outline-offset:1px; background:#fff; }

  .feedback-badge{
    display:none; margin-top:10px; font-family:var(--mono); font-size:12.5px;
    padding:8px 10px; border-radius:3px; border:1px solid var(--rule);
  }
  .feedback-badge.show{ display:block; }
  .feedback-badge.correct{ background:#e4ede9; border-color:var(--good); }
  .feedback-badge.incorrect{ background:#fbeae6; border-color:var(--warn); }

  .submit-bar{ position:sticky; bottom:0; background:linear-gradient(to top, var(--paper) 60%, transparent); padding:20px 0 0; margin-top:36px; }
  #submitBtn{
    width:100%; padding:16px; font-family:var(--sans); font-weight:600; font-size:16px;
    color:#fff; background:var(--accent); border:none; border-radius:4px; cursor:pointer;
  }
  #submitBtn:disabled{ background:#8a9791; cursor:not-allowed; }
  #submitBtn:hover:not(:disabled){ background:#264d42; }
  #statusLine{ font-family:var(--mono); font-size:13px; text-align:center; margin-top:10px; min-height:18px; color:var(--warn); }
  #statusLine.ok{ color:var(--accent); }

  .result-panel{ display:none; margin-top:28px; background:var(--card); border:1px solid var(--rule); border-radius:4px; padding:28px; text-align:center; }
  .result-panel.show{ display:block; }
  .result-score{ font-family:var(--serif); font-size:52px; margin:6px 0; color:var(--accent); }
  .result-breakdown{ font-family:var(--mono); font-size:13px; color:#4a504a; margin-top:8px; }
  .result-panel p{ color:#4a504a; margin:6px 0 0; }
</style>
</head>
<body>

<div id="timerBar">
  <span id="progressText">Question 0 / 0 answered</span>
  <span id="timeLeft">45:00</span>
</div>

<header>
  <p class="eyebrow">45-Minute Certification Exam · Self-Scored</p>
  <h1>Claude &amp; Claude Code — Certification</h1>
  <p>25 multiple-choice questions (auto-scored instantly) plus 5 hands-on tasks (graded from what you actually
     pasted). You have 45 minutes. Submit once at the bottom — your score, per-question feedback, and the
     correct answers are shown immediately, and everything is logged to the team sheet.</p>
</header>

<main>
  <form id="examForm">
    <div class="identity">
      <div class="field">
        <label for="empName">Name</label>
        <input type="text" id="empName" name="empName" required autocomplete="name">
      </div>
      <div class="field">
        <label for="empEmail">Work Email</label>
        <input type="email" id="empEmail" name="empEmail" required autocomplete="email">
      </div>
    </div>

    <div class="section-header">Part A — Multiple Choice</div>
    <p class="section-sub" id="mcqSub">Loading…</p>
    <div id="mcqContainer"></div>

    <div class="section-header">Part B — Hands-On Tasks</div>
    <p class="section-sub" id="practicalSub">Do each one for real in Claude, then paste the actual output.</p>
    <div id="practicalContainer"></div>

    <div class="submit-bar">
      <button type="submit" id="submitBtn">Submit &amp; Get Certified</button>
      <div id="statusLine"></div>
    </div>
  </form>

  <div class="result-panel" id="resultPanel">
    <p class="eyebrow" style="margin:0;">Your Result</p>
    <div class="result-score" id="resultScore">–</div>
    <p id="resultVerdict"></p>
    <div class="result-breakdown" id="resultBreakdown"></div>
  </div>
</main>

<script>
  const API_BASE = "";
  let EXAM = null;
  let timerInterval = null;
  let secondsLeft = 0;
  let submitted = false;

  async function loadExam(){
    const res = await fetch(`${API_BASE}/api/exam`);
    EXAM = await res.json();
    renderExam();
    startTimer(EXAM.duration_minutes * 60);
  }

  function renderExam(){
    document.getElementById('mcqSub').textContent =
      `${EXAM.mcqs.length} questions · ${EXAM.mcq_total} points`;
    document.getElementById('practicalSub').textContent =
      `${EXAM.practicals.length} tasks · ${EXAM.practical_total} points · do each for real in Claude, then paste the actual output`;

    const mcqContainer = document.getElementById('mcqContainer');
    EXAM.mcqs.forEach((q, idx) => {
      const card = document.createElement('div');
      card.className = 'q-card';
      const optionsHtml = q.options.map((opt, i) => `
        <label>
          <input type="radio" name="mcq-${q.id}" value="${i}" onchange="updateProgress()">
          <span>${escapeHtml(opt)}</span>
        </label>
      `).join('');
      card.innerHTML = `
        <div class="q-meta">Q${idx+1} · ${escapeHtml(q.source)} · ${q.points} pts</div>
        <div class="q-text">${escapeHtml(q.question)}</div>
        <div class="options">${optionsHtml}</div>
        <div class="feedback-badge" id="mcq-badge-${q.id}"></div>
      `;
      mcqContainer.appendChild(card);
    });

    const practicalContainer = document.getElementById('practicalContainer');
    EXAM.practicals.forEach((p, idx) => {
      const card = document.createElement('div');
      card.className = 'task-card';
      card.innerHTML = `
        <div class="q-meta">Task ${idx+1} · ${escapeHtml(p.source)} · ${p.max_score} pts</div>
        <div class="task-title">${escapeHtml(p.title)}</div>
        <div class="task-prompt">${escapeHtml(p.prompt_given)}</div>
        <textarea id="practical-${p.id}" placeholder="Paste what Claude actually gave you…" oninput="updateProgress()"></textarea>
        <div class="feedback-badge" id="practical-badge-${p.id}"></div>
      `;
      practicalContainer.appendChild(card);
    });

    updateProgress();
  }

  function updateProgress(){
    if (!EXAM) return;
    let answered = 0;
    const totalItems = EXAM.mcqs.length + EXAM.practicals.length;
    EXAM.mcqs.forEach(q => {
      if (document.querySelector(`input[name="mcq-${q.id}"]:checked`)) answered++;
    });
    EXAM.practicals.forEach(p => {
      const el = document.getElementById(`practical-${p.id}`);
      if (el && el.value.trim()) answered++;
    });
    document.getElementById('progressText').textContent = `${answered} / ${totalItems} answered`;
  }

  function startTimer(totalSeconds){
    secondsLeft = totalSeconds;
    updateTimerDisplay();
    timerInterval = setInterval(() => {
      secondsLeft--;
      updateTimerDisplay();
      if (secondsLeft <= 300) document.getElementById('timerBar').classList.add('warn');
      if (secondsLeft <= 0){
        clearInterval(timerInterval);
        if (!submitted){
          document.getElementById('statusLine').textContent = "Time's up — submitting automatically.";
          document.getElementById('examForm').requestSubmit();
        }
      }
    }, 1000);
  }

  function updateTimerDisplay(){
    const m = Math.max(0, Math.floor(secondsLeft / 60));
    const s = Math.max(0, secondsLeft % 60);
    document.getElementById('timeLeft').textContent = `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
  }

  function escapeHtml(str){
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  document.getElementById('examForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (submitted) return;
    const btn = document.getElementById('submitBtn');
    const statusLine = document.getElementById('statusLine');
    const name = document.getElementById('empName').value.trim();
    const email = document.getElementById('empEmail').value.trim();

    if (!name || !email){
      statusLine.textContent = 'Please enter your name and email.';
      statusLine.className = '';
      return;
    }

    const mcq_answers = {};
    EXAM.mcqs.forEach(q => {
      const checked = document.querySelector(`input[name="mcq-${q.id}"]:checked`);
      if (checked) mcq_answers[q.id] = parseInt(checked.value, 10);
    });
    const practical_answers = {};
    EXAM.practicals.forEach(p => {
      const el = document.getElementById(`practical-${p.id}`);
      practical_answers[p.id] = el ? el.value : '';
    });

    submitted = true;
    clearInterval(timerInterval);
    btn.disabled = true;
    btn.textContent = 'Grading…';
    statusLine.textContent = 'Scoring MCQs instantly, grading hands-on tasks — this can take a moment.';
    statusLine.className = '';

    try{
      const res = await fetch(`${API_BASE}/api/submit`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name, email, mcq_answers, practical_answers})
      });
      const data = await res.json();

      if (!res.ok){
        statusLine.textContent = data.error || 'Something went wrong.';
        statusLine.className = '';
        btn.disabled = false;
        btn.textContent = 'Submit & Get Certified';
        submitted = false;
        return;
      }

      EXAM.mcqs.forEach(q => {
        const r = data.mcq_results[q.id];
        const badge = document.getElementById(`mcq-badge-${q.id}`);
        if (r && badge){
          badge.classList.add('show', r.correct ? 'correct' : 'incorrect');
          badge.innerHTML = r.correct
            ? `✓ Correct (+${r.points} pts)`
            : `✗ Incorrect — correct answer: ${escapeHtml(r.correct_answer)}`;
        }
      });

      EXAM.practicals.forEach(p => {
        const r = data.practical_results[p.id];
        const badge = document.getElementById(`practical-badge-${p.id}`);
        if (r && badge){
          badge.classList.add('show', r.score >= p.max_score * 0.6 ? 'correct' : 'incorrect');
          badge.innerHTML = `<b>${r.score}/${p.max_score}</b> — ${escapeHtml(r.feedback)}`;
        }
      });

      document.getElementById('resultScore').textContent = `${data.total_score} / ${data.grand_total}`;
      document.getElementById('resultVerdict').textContent = `${data.percentage}% — ${data.overall_feedback}`;
      document.getElementById('resultBreakdown').textContent =
        `MCQ: ${data.mcq_score}/${data.mcq_total} · Hands-on: ${data.practical_score}/${data.practical_total}` +
        (data.sheet_status === 'ok' ? ' · logged to the team sheet' :
         data.sheet_status === 'not_configured' ? '' : ' · could not log to the sheet, ask your admin');
      document.getElementById('resultPanel').classList.add('show');
      document.getElementById('resultPanel').scrollIntoView({behavior:'smooth'});

      statusLine.textContent = 'Done.';
      statusLine.className = 'ok';
      btn.textContent = 'Submitted';
    } catch(err){
      statusLine.textContent = 'Network error — please screenshot this and reach out.';
      statusLine.className = '';
      btn.disabled = false;
      btn.textContent = 'Submit & Get Certified';
      submitted = false;
    }
  });

  loadExam();
</script>

</body>
</html>
"""

app = Flask(__name__)
CORS(app)

SHEET_WEBHOOK_URL = os.environ.get("SHEET_WEBHOOK_URL", "").strip()


def _public_mcqs() -> List[dict]:
    """Strips `correct` before sending questions to the browser."""
    out = []
    for q in MCQS:
        clean = {k: v for k, v in q.items() if k != "correct"}
        out.append(clean)
    return out


@app.route("/")
def serve_index():
    return app.response_class(INDEX_HTML, mimetype="text/html")


@app.route("/api/exam")
def api_exam():
    return jsonify({
        "mcqs": _public_mcqs(),
        "practicals": PRACTICALS,
        "duration_minutes": EXAM_DURATION_MINUTES,
        "mcq_total": MCQ_TOTAL,
        "practical_total": PRACTICAL_TOTAL,
        "grand_total": GRAND_TOTAL,
    })


@app.route("/api/submit", methods=["POST"])
def api_submit():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    mcq_answers = data.get("mcq_answers") or {}
    practical_answers = data.get("practical_answers") or {}

    if not name or not email:
        return jsonify({"error": "Name and email are required."}), 400
    if "@" not in email:
        return jsonify({"error": "Please enter a valid email."}), 400

    try:
        result = grade_exam(name, email, mcq_answers, practical_answers)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Grading failed: {exc}"}), 500

    row_status = "not_configured"
    if SHEET_WEBHOOK_URL:
        row_status = _push_to_sheet(result)

    return jsonify({
        "mcq_results": result["mcq_results"],
        "practical_results": result["practical_results"],
        "mcq_score": result["mcq_score"],
        "mcq_total": MCQ_TOTAL,
        "practical_score": result["practical_score"],
        "practical_total": PRACTICAL_TOTAL,
        "total_score": result["total_score"],
        "grand_total": GRAND_TOTAL,
        "percentage": result["percentage"],
        "overall_feedback": result["overall_feedback"],
        "sheet_status": row_status,
    })


def _push_to_sheet(result: dict) -> str:
    row = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "name": result["name"],
        "email": result["email"],
        "mcq_score": result["mcq_score"],
        "mcq_total": MCQ_TOTAL,
        "practical_score": result["practical_score"],
        "practical_total": PRACTICAL_TOTAL,
        "total_score": result["total_score"],
        "grand_total": GRAND_TOTAL,
        "percentage": result["percentage"],
        "overall_feedback": result["overall_feedback"],
    }
    for p in PRACTICALS:
        pr = result["practical_results"].get(p["id"], {"score": 0, "feedback": ""})
        row[f"{p['id']}_score"] = pr["score"]
        row[f"{p['id']}_feedback"] = pr["feedback"]

    mcq_correct_count = sum(1 for r in result["mcq_results"].values() if r["correct"])
    row["mcq_correct_count"] = mcq_correct_count
    row["mcq_question_count"] = len(MCQS)

    try:
        resp = requests.post(SHEET_WEBHOOK_URL, json=row, timeout=15)
        if resp.status_code == 200:
            return "ok"
        return f"sheet_error_{resp.status_code}"
    except requests.RequestException as exc:
        return f"sheet_error_{exc}"
