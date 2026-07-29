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
        "id": "m1", "type": "mcq", "source": "Prompt Anatomy",
        "question": "Two engineers ask for the same endpoint. Which prompt is far more likely to come back correct on the first try?",
        "options": [
            "\"Add an endpoint for orders.\"",
            "\"Add an orders endpoint — you know what I mean, make it good.\"",
            "\"Add orders. Use best practices and modern standards.\"",
            "\"Add GET /api/orders in api/routes/orders.ts, following the pagination + auth pattern already in users.ts. Return 200 {orders: Order[], nextCursor}, 401 when unauthenticated. Don't touch the shared error middleware. Add a test alongside the existing route tests.\"",
        ],
        "correct": 3, "points": 2,
    },
    {
        "id": "m2", "type": "mcq", "source": "Prompt Anatomy",
        "question": "Which of these is NOT one of the parts of a strong engineering prompt?",
        "options": [
            "Task — the single change you want",
            "Context — the files, patterns and data shapes it must fit",
            "Constraints — what it must not touch or break",
            "Deadline — when you need it by",
        ],
        "correct": 3, "points": 2,
    },
    {
        "id": "m3", "type": "mcq", "source": "Prompt Anatomy",
        "question": "What belongs in the ACCEPTANCE line of a prompt spec?",
        "options": [
            "The observable behaviour that proves it's done — inputs, outputs, UI states and error cases",
            "A guess at how long the change will take",
            "The name of the person who requested the ticket",
            "A reminder to write clean code",
        ],
        "correct": 0, "points": 2,
    },
    {
        "id": "m4", "type": "mcq", "source": "Prompt Anatomy",
        "question": "\"Refactor the auth module, add role-based permissions, and redo the settings page\" in one prompt. What's the main cost?",
        "options": [
            "It uses more tokens than three separate prompts",
            "Nothing — bigger prompts always produce better architecture",
            "One giant diff that's hard to review, hard to test and impossible to bisect when something breaks — split it into separate prompts, each with its own verification",
            "The model will refuse to answer prompts with three verbs",
        ],
        "correct": 2, "points": 2,
    },
    {
        "id": "m5", "type": "mcq", "source": "Prompt Anatomy",
        "question": "Which single constraint most reliably prevents an agent from breaking things you didn't ask it to touch?",
        "options": [
            "\"Be careful.\"",
            "\"Write good code.\"",
            "Naming what must NOT change — the public API shape, the DB schema, the passing tests, the shared components",
            "\"Use TypeScript.\"",
        ],
        "correct": 2, "points": 2,
    },
    {
        "id": "m6", "type": "mcq", "source": "Context & Conventions",
        "question": "The model keeps inventing field names for an API your backend already defines. Fastest fix?",
        "options": [
            "Paste the real schema / type definition — or point at the exact file — so it works from the contract instead of guessing",
            "Ask it to be more careful next time",
            "Lower the temperature",
            "Ask the same question again in a new session",
        ],
        "correct": 0, "points": 2,
    },
    {
        "id": "m7", "type": "mcq", "source": "Context & Conventions",
        "question": "What is CLAUDE.md for?",
        "options": [
            "A changelog of every past session",
            "A list of banned commands",
            "A file read automatically at the start of every session, so your stack, conventions, folder layout and test/lint commands never have to be retyped",
            "The project's README for end users",
        ],
        "correct": 2, "points": 2,
    },
    {
        "id": "m8", "type": "mcq", "source": "Context & Conventions",
        "question": "Your app has a design-system <Button>, but the agent keeps hand-rolling new styled buttons. What instruction stops the duplication?",
        "options": [
            "\"Don't write CSS.\"",
            "\"Make the button look nicer.\"",
            "\"Use fewer files.\"",
            "\"Before building any UI, search the codebase for an existing component or util that already does this and reuse or extend it — only create something new if nothing fits, and say why.\"",
        ],
        "correct": 3, "points": 2,
    },
    {
        "id": "m9", "type": "mcq", "source": "Context & Conventions",
        "question": "You're deep into the same ticket and the context bar is climbing. You then need to switch to a totally unrelated task. What's the right sequence?",
        "options": [
            "/clear now, then /compact later",
            "/compact for the climbing context on the same ticket, /clear when you switch to the unrelated task",
            "/init both times",
            "Neither — just keep going, context never needs managing",
        ],
        "correct": 1, "points": 2,
    },
    {
        "id": "m10", "type": "mcq", "source": "Context & Conventions",
        "question": "The \"kill repetition\" rule says a retyped instruction belongs in a saved command or Skill once you've typed a version of it:",
        "options": [
            "Once",
            "More than twice in a week",
            "Only if a teammate asks",
            "Every single session",
        ],
        "correct": 1, "points": 2,
    },
    {
        "id": "m11", "type": "mcq", "source": "Plan Before Edits",
        "question": "Which command switches Claude Code into a mode where it proposes an approach but edits nothing until you approve?",
        "options": [
            "/compact",
            "/clear",
            "/plan (or Shift+Tab)",
            "/init",
        ],
        "correct": 2, "points": 2,
    },
    {
        "id": "m12", "type": "mcq", "source": "Plan Before Edits",
        "question": "When is it worth spending an extra minute asking for two or three approaches with trade-offs before any code is written?",
        "options": [
            "When the change is hard to reverse — data model, auth, state management, API contracts, anything other code will be built on top of",
            "On every change, including a one-line CSS tweak",
            "Never — it just slows delivery down",
            "Only when you personally don't know the language",
        ],
        "correct": 0, "points": 2,
    },
    {
        "id": "m13", "type": "mcq", "source": "Plan Before Edits",
        "question": "The ticket is under-specified and the agent starts coding anyway on guessed requirements. Which addition to your prompt prevents the wasted round-trip?",
        "options": [
            "\"Work faster.\"",
            "\"Write more comments.\"",
            "\"Use a bigger model.\"",
            "\"Before writing any code, list your assumptions and any question whose answer would change the design.\"",
        ],
        "correct": 3, "points": 2,
    },
    {
        "id": "m14", "type": "mcq", "source": "Debugging Prompts",
        "question": "\"The page is broken, it won't load, please fix.\" What is most missing from this bug prompt?",
        "options": [
            "Politeness",
            "The exact error text and stack trace, the steps that reproduce it, and what you expected to happen instead",
            "The name of the framework's founder",
            "A deadline",
        ],
        "correct": 1, "points": 2,
    },
    {
        "id": "m15", "type": "mcq", "source": "Debugging Prompts",
        "question": "A bug lands in your queue. What should the prompt ask for FIRST?",
        "options": [
            "A reliable reproduction, ideally a failing test — a fix you can't reproduce is a fix you can't verify",
            "The patch, immediately — speed matters most",
            "A rewrite of the whole module",
            "A list of every file in the repo",
        ],
        "correct": 0, "points": 2,
    },
    {
        "id": "m16", "type": "mcq", "source": "Debugging Prompts",
        "question": "Why ask \"explain the root cause before you patch it\" instead of just \"fix it\"?",
        "options": [
            "It makes the response longer, which is always better",
            "The model can't write code and explanations in the same reply",
            "It reduces token cost",
            "A patch aimed at the symptom often just relocates the bug; the cause tells you whether other call sites have the same defect",
        ],
        "correct": 3, "points": 2,
    },
    {
        "id": "m17", "type": "mcq", "source": "Debugging Prompts",
        "question": "Third failed attempt at the same bug, in a thread that's now enormous. Best next move?",
        "options": [
            "Start a fresh session with only the current code, the failing test and the real error — a thread full of wrong turns keeps steering toward them",
            "Keep going in the same thread — it has the most context",
            "Delete the failing test so CI is green",
            "Ask the same question louder, in caps",
        ],
        "correct": 0, "points": 2,
    },
    {
        "id": "m18", "type": "mcq", "source": "Verify, Don't Trust",
        "question": "\"Ask for proof, not a claim\" after a feature that writes to the database should request:",
        "options": [
            "A confident one-line summary that it works",
            "A screenshot of the code",
            "The actual test command and its output, the row queried back for real, the real request/response, and any step that failed silently",
            "Nothing — trust the agent's word",
        ],
        "correct": 2, "points": 2,
    },
    {
        "id": "m19", "type": "mcq", "source": "Verify, Don't Trust",
        "question": "The agent reports \"all tests pass\" but pasted no output. What's the correct response?",
        "options": [
            "Merge it — it said so",
            "Re-run the prompt with different wording",
            "Add more tests without checking the existing ones",
            "Ask for the exact command it ran and its unedited output; an unverified claim is not evidence",
        ],
        "correct": 3, "points": 2,
    },
    {
        "id": "m20", "type": "mcq", "source": "Verify, Don't Trust",
        "question": "Which prompt pattern cuts repeat regressions the most on a bug fix?",
        "options": [
            "\"Write a failing test that reproduces this bug, show me it failing, then make it pass without weakening the assertions, and show the full suite green.\"",
            "\"Fix it and don't break anything.\"",
            "\"Fix it, then delete any test that turns red.\"",
            "\"Fix it and add a comment explaining the fix.\"",
        ],
        "correct": 0, "points": 2,
    },
    {
        "id": "m21", "type": "mcq", "source": "Verify, Don't Trust",
        "question": "What counts as \"done\" for an agent-written UI change?",
        "options": [
            "The diff looks reasonable when you skim it",
            "The app was actually run and the changed screen observed — the happy path plus loading, empty and error states, with a clean console",
            "The agent said it followed best practices",
            "It compiles",
        ],
        "correct": 1, "points": 2,
    },
    {
        "id": "m22", "type": "mcq", "source": "Review & Regression",
        "question": "In the writer/reviewer pattern, what should you paste into the fresh review session?",
        "options": [
            "The whole conversation history from the writing session",
            "Only the diff — no backstory",
            "A summary of what you think is wrong",
            "Nothing — just ask it to guess",
        ],
        "correct": 1, "points": 2,
    },
    {
        "id": "m23", "type": "mcq", "source": "Review & Regression",
        "question": "Which review prompt produces the most actionable output?",
        "options": [
            "\"Review this diff for correctness, unhandled error paths, and regressions to existing callers. Rank findings by severity and, for each, give the failing input that triggers it.\"",
            "\"Is this code good?\"",
            "\"Any thoughts?\"",
            "\"Rate this out of 10.\"",
        ],
        "correct": 0, "points": 2,
    },
    {
        "id": "m24", "type": "mcq", "source": "Web App Craft",
        "question": "Which prompt habit most reduces the flood of \"it broke when…\" tickets after a UI ship?",
        "options": [
            "Asking for prettier animations",
            "Asking the model to add more comments",
            "Asking for fewer files",
            "Specifying the non-happy paths up front — loading, empty, error, offline, slow network, and unauthorised — as part of the acceptance criteria",
        ],
        "correct": 3, "points": 2,
    },
    {
        "id": "m25", "type": "mcq", "source": "Web App Craft",
        "question": "\"Make the dashboard faster.\" How should you re-write it?",
        "options": [
            "\"Make the dashboard much faster, it's really important.\"",
            "\"Rewrite the dashboard in a faster framework.\"",
            "\"LCP is 4.2s on a mid-tier phone; target is under 2.5s. Profile first, tell me what actually dominates, then change only that — and show me the before/after numbers.\"",
            "\"Add caching everywhere.\"",
        ],
        "correct": 2, "points": 2,
    },
]
PRACTICALS = [
    {
        "id": "p1", "type": "practical", "source": "Prompt Anatomy",
        "title": "Turn a Vague Ticket Into a Specced Prompt",
        "prompt_given": "In Claude, rewrite this ticket as a prompt with TASK / CONTEXT / CONSTRAINTS / ACCEPTANCE / "
                         "FORMAT lines: \"The checkout page is slow and sometimes double-charges people, please fix it.\" "
                         "The ACCEPTANCE line must name observable behaviour (including the error and loading states), and "
                         "CONSTRAINTS must name what may not change. Paste Claude's actual response below.",
        "look_for": "Five labelled lines (TASK, CONTEXT, CONSTRAINTS, ACCEPTANCE, FORMAT), each specific to checkout "
                    "latency and double-charging. ACCEPTANCE must state checkable behaviour — e.g. a repeat submit "
                    "produces exactly one charge, plus loading/error states — not vague goals like 'make it fast'. "
                    "CONSTRAINTS must name concrete off-limits surfaces (payment provider contract, existing tests, DB "
                    "schema). Nothing important left for Claude to guess.",
        "max_score": 10,
    },
    {
        "id": "p2", "type": "practical", "source": "Plan Before Edits",
        "title": "Plan Mode on a Real Feature",
        "prompt_given": "Pick a real feature you actually need in your web app. In Claude Code, enter plan mode "
                         "(Shift+Tab or /plan) and ask for two viable approaches with trade-offs, the exact files each "
                         "would touch, the riskiest step, and how you'd roll it back — with NO edits made. Paste the "
                         "actual plan, and one line on which approach you picked and why.",
        "look_for": "A real plan produced with no code written: two distinct approaches with honest trade-offs, named "
                    "files/paths from an actual codebase, an explicit riskiest step and a rollback path, plus the "
                    "employee's own pick and reasoning. Generic advice with no real file names, or a plan that already "
                    "contains applied edits, scores low.",
        "max_score": 10,
    },
    {
        "id": "p3", "type": "practical", "source": "Debugging Prompts",
        "title": "Reproduce Before You Fix",
        "prompt_given": "Take a real bug in your project. Prompt Claude with the exact error text, the repro steps and "
                         "the expected behaviour, and ask it to (1) write a failing test that reproduces the bug, (2) "
                         "explain the root cause, (3) only then fix it without weakening the test. Paste the failing test "
                         "output, the root-cause explanation, and the passing output after the fix.",
        "look_for": "Evidence of the real sequence: a genuine failing test run (red output), a root-cause explanation "
                    "that names the actual defect rather than restating the symptom, and a subsequent passing run. A fix "
                    "with no failing-test-first evidence, or output that looks asserted rather than pasted, scores low.",
        "max_score": 10,
    },
    {
        "id": "p4", "type": "practical", "source": "Review & Regression",
        "title": "Reviewer Pass on Your Own Diff",
        "prompt_given": "Open a FRESH Claude session and paste only your current diff (git diff) — no backstory. Ask: "
                         "\"Review this diff for correctness, unhandled error paths and regressions to existing callers. "
                         "Rank findings by severity and give the failing input that triggers each one.\" Paste the "
                         "findings, then note which ones you actually fixed and which you consciously dismissed.",
        "look_for": "A real diff was reviewed in a clean session: severity-ranked findings tied to concrete inputs or "
                    "call sites, plus the employee's own triage of what was fixed vs dismissed and why. Generic review "
                    "checklists with no reference to actual changed code score low.",
        "max_score": 10,
    },
    {
        "id": "p5", "type": "practical", "source": "Verify, Don't Trust",
        "title": "Ask for Proof, Not a Claim",
        "prompt_given": "On a real change that writes to a database or calls an external service, append to your prompt: "
                         "\"When done, show me: 1) the exact test command and its unedited output, 2) the row it wrote, "
                         "queried back for real, 3) the actual request/response, 4) any step that failed silently.\" "
                         "Paste Claude's actual evidence-backed response below.",
        "look_for": "Real, checkable output — an actual command with its output, a genuinely queried-back row or real "
                    "request/response — not an assertion that it works. Bonus credit if a silently failing step was "
                    "surfaced and dealt with.",
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
    "You are a strict but fair engineering trainer grading how well an employee PROMPTED Claude / Claude Code "
    "while building or fixing a real web app. The skill being tested is prompting that ships working software "
    "faster with fewer bugs and fewer follow-up fixes: specific asks, real context, explicit constraints, "
    "checkable acceptance criteria, planning before edits, reproducing before fixing, and demanding evidence "
    "instead of accepting claims. You will be given the task instructions, the criteria the trainer is looking "
    "for, and the employee's pasted output (their transcript, notes, or result). "
    "Score ONLY what is shown — do not assume steps happened if they weren't described or pasted. "
    "Reward pasted evidence from a real codebase (real file paths, real command output, real errors); "
    "mark down generic textbook answers that could have been written without touching any project. "
    "Respond with STRICT JSON only, no markdown, in this exact shape: "
    '{"score": <integer 0-10>, "feedback": "<one or two sentence, specific, constructive feedback>"}'
)



# ---------------------------------------------------------------------------
# DISTRIBUTED RATE LIMITER (Upstash Redis, optional — no-op if not configured)
#
# Cerebras's free tier is ~5 requests/minute, SHARED across every employee
# hitting this app at once, not per-user. A stateless Vercel function can't
# coordinate that on its own — two people submitting at the same moment run
# in two separate function instances with no shared memory. This uses
# Upstash Redis's plain REST API (just `requests`, no extra package) as the
# one piece of shared state everyone's requests check against.
#
# Setup (one click, inside Vercel itself, no separate signup):
#   Vercel project -> Storage tab -> Create Database -> Upstash Redis (free)
#   -> Connect to this project. Vercel auto-injects either:
#     UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN, or
#     KV_REST_API_URL / KV_REST_API_TOKEN
#   This checks for both names. If neither is set, every call below is a
#   safe no-op and the app works exactly as before (just without
#   cross-request protection).
# ---------------------------------------------------------------------------

CEREBRAS_SAFE_RPM = 4  # stay under Cerebras's ~5 req/min free-tier cap, leave a margin
RATE_WINDOW_SECONDS = 60
RATE_KEY = "examapp:cerebras_rpm"


def _redis_config():
    url = os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("KV_REST_API_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN") or os.environ.get("KV_REST_API_TOKEN")
    if url and token:
        return url.rstrip("/"), token
    return None, None


def _redis_get(url, token, path):
    resp = requests.get(f"{url}/{path}", headers={"Authorization": f"Bearer {token}"}, timeout=5)
    resp.raise_for_status()
    return resp.json().get("result")


def acquire_cerebras_slot(max_wait_seconds: int = 90) -> bool:
    """Blocks (sleeping) until it's safe to make one Cerebras call, using a shared
    Redis counter that resets every RATE_WINDOW_SECONDS. Returns True once a slot
    is acquired (or immediately if Redis isn't configured), or False if
    max_wait_seconds elapses first — the caller still attempts the call either
    way, since the per-call 429 retry is the final safety net."""
    url, token = _redis_config()
    if not url:
        return True

    deadline = time.time() + max_wait_seconds
    while time.time() < deadline:
        try:
            current = _redis_get(url, token, f"get/{RATE_KEY}")
            current = int(current) if current is not None else 0

            if current < CEREBRAS_SAFE_RPM:
                new_val = _redis_get(url, token, f"incr/{RATE_KEY}")
                if int(new_val) == 1:
                    _redis_get(url, token, f"expire/{RATE_KEY}/{RATE_WINDOW_SECONDS}")
                return True

            ttl = _redis_get(url, token, f"ttl/{RATE_KEY}")
            wait_for = (int(ttl) + 1) if (ttl and int(ttl) > 0) else 12
            wait_for = min(wait_for, max(1, deadline - time.time()))
            time.sleep(wait_for)
        except requests.RequestException:
            return True  # Redis unreachable this attempt — don't block grading entirely

    return False


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
            acquire_cerebras_slot()
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
        verdict = ("Certified — prompts are specific, constrained and evidence-backed. This is the habit set that "
                   "ships web app work fast and keeps it from coming back as bugs.")
    elif state["percentage"] >= 70:
        verdict = "Passed — solid prompting habits, a few areas worth tightening (check the per-question feedback)."
    elif state["percentage"] >= 50:
        verdict = ("Borderline — the basics are there, but too much is still being left for the model to guess. "
                   "Review the missed sections and retake.")
    else:
        verdict = ("Not yet passing — prompts are still under-specified and under-verified. Re-read the source "
                   "material on spec, constraints, and proof-over-claims, then retake.")
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
<title>Prompting for Faster, Less Buggy Web Apps — Certification Exam</title>
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
  <p class="eyebrow">45-Minute Certification Exam · Software Engineering</p>
  <h1>Prompting for Faster, Less Buggy Web Apps</h1>
  <p>Every question here is about one thing: prompting Claude so web app work ships faster and comes back as
     bugs less often — specific asks, real context, explicit constraints, checkable acceptance criteria,
     planning before edits, reproducing before fixing, and proof instead of claims.
     25 multiple-choice questions (auto-scored instantly) plus 5 hands-on tasks you do for real in your own
     project (graded from what you actually pasted). You have 45 minutes. Submit once at the bottom — your
     score, per-question feedback, and the correct answers are shown immediately, and everything is logged to
     the team sheet.</p>
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
            # Apps Script can return HTTP 200 with an HTML "authorization required" or
            # error page instead of our expected JSON — catch that case too, not just
            # non-200 statuses, since a silent 200 would otherwise look like success.
            try:
                body = resp.json()
                if body.get("status") == "ok":
                    return "ok"
                return f"sheet_error_unexpected_response_{str(body)[:150]}"
            except ValueError:
                return f"sheet_error_non_json_200_{resp.text[:150]}"
        return f"sheet_error_{resp.status_code}_{resp.text[:150]}"
    except requests.RequestException as exc:
        return f"sheet_error_{exc}"
