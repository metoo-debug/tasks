"""
Vercel Python serverless entrypoint — everything in ONE file on purpose.

Vercel's Python runtime doesn't reliably bundle sibling .py files next to the
entrypoint, so tasks.py/grader.py living alongside this file caused
"ModuleNotFoundError: No module named 'tasks'" in production even though they
worked locally. Merging everything into api/index.py removes that failure mode
entirely — there's nothing left to fail to bundle.
"""

import os
import json
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# TASKS + RUBRIC (was api/tasks.py)
# ---------------------------------------------------------------------------

TASKS = [
    {
        "id": 1, "part": "Prompting & Planning",
        "title": "Turn a Vague Ask Into a Tagged Prompt",
        "prompt_given": 'Rewrite this using task, context, constraints, and format: "Can you add caching to the search endpoint, '
                         'it\'s kind of slow, don\'t break the existing tests"',
        "look_for": "A response with the request clearly split into task / context / constraints / format sections — "
                    "nothing left for Claude to guess at.",
        "max_score": 5,
    },
    {
        "id": 2, "part": "Prompting & Planning",
        "title": "Fill a 5-Line Spec Block Before You Build",
        "prompt_given": "Write a TASK / DATA / SERVICES / CONSTRAINTS / FORMAT block for: when a user cancels a subscription, "
                         "write a row to a cancellations table and call the billing provider's refund API for partial-month refunds.",
        "look_for": "Five short labeled lines (TASK, DATA, SERVICES, CONSTRAINTS, FORMAT), each substantive and specific "
                    "to the cancellation/refund scenario.",
        "max_score": 5,
    },
    {
        "id": 3, "part": "Prompting & Planning",
        "title": "Force Plan Mode Before Anything Gets Edited",
        "prompt_given": "Don't write any code yet. First give me a plan: files touched, approach, and risks. I'll approve it, "
                         "then you implement. (or Shift+Tab into Plan Mode)",
        "look_for": "Evidence Claude proposed a plan (files touched, approach, risks) and paused for approval before editing anything.",
        "max_score": 5,
    },
    {
        "id": 4, "part": "Prompting & Planning",
        "title": "Start Minimal, Then Correct Instead of Front-Loading",
        "prompt_given": "I need to debug why our token refresh sometimes fails silently. Start by asking me whatever you need "
                         "to know — don't assume the stack yet.",
        "look_for": "Claude asked a short, targeted clarifying question back instead of guessing the architecture.",
        "max_score": 5,
    },
    {
        "id": 5, "part": "Context & Cost Management",
        "title": "Generate Your Memory File, Then Trim It",
        "prompt_given": "/init",
        "look_for": "A CLAUDE.md was generated reflecting the real project stack/conventions, and the submitter describes "
                    "trimming anything not actively true.",
        "max_score": 5,
    },
    {
        "id": 6, "part": "Context & Cost Management",
        "title": "Use /compact and /clear for What They're Actually For",
        "prompt_given": "/compact (mid-task, same ticket) and /clear (switching to an unrelated task)",
        "look_for": "Evidence /compact shrank context while keeping task memory, and /clear was used specifically when "
                    "switching to unrelated work.",
        "max_score": 5,
    },
    {
        "id": 7, "part": "Context & Cost Management",
        "title": "Check What a Session Actually Costs",
        "prompt_given": "/cost",
        "look_for": "A real cost/usage breakdown for the session was retrieved and reported.",
        "max_score": 5,
    },
    {
        "id": 8, "part": "Context & Cost Management",
        "title": "Run a Second Task in Its Own Worktree",
        "prompt_given": "git worktree add ../my-repo-hotfix -b hotfix/quick-fix && cd ../my-repo-hotfix && claude",
        "look_for": "A second, isolated Claude Code session running on its own branch/worktree, separate from the main session.",
        "max_score": 5,
    },
    {
        "id": 9, "part": "Verification & Delegation",
        "title": "Ask for Proof, Not a Claim",
        "prompt_given": "When done, show me: 1) actual test output, 2) the row it wrote, queried back for real, 3) any step "
                         "that failed silently.",
        "look_for": "Real, queryable output/evidence was returned — not just an assertion that something works.",
        "max_score": 5,
    },
    {
        "id": 10, "part": "Verification & Delegation",
        "title": "Get a Genuinely Fresh Review",
        "prompt_given": "/clear, then paste only the diff: Review this diff for security, edge cases, and readability. "
                         "Report issues — don't rewrite it.",
        "look_for": "A fresh-context review (post /clear) surfaced issues the original session had missed, without rewriting the code.",
        "max_score": 5,
    },
    {
        "id": 11, "part": "Verification & Delegation",
        "title": "Delegate Research to a Subagent",
        "prompt_given": "Use a subagent to investigate how our authentication handles token refresh, and whether we already "
                         "have an OAuth utility to reuse. Report back a summary, not the raw files.",
        "look_for": "A concise summary came back from a subagent, with the main session's context kept clean of raw file contents.",
        "max_score": 5,
    },
    {
        "id": 12, "part": "Verification & Delegation",
        "title": "Turn Your Most Repeated Prompt Into a Command",
        "prompt_given": "Create .claude/commands/int-test.md with a description and body that writes a real integration test "
                         "for $ARGUMENTS hitting the endpoint and confirming the DB row.",
        "look_for": "A working slash command file was created, and invoking /int-test <arg> runs the full instruction with zero retyping.",
        "max_score": 5,
    },
    {
        "id": 13, "part": "Connectors, MCP & RAG",
        "title": "Turn On One Real Connector",
        "prompt_given": "Settings → Connectors → Connect (GitHub or Google Drive) → approve. Then ask: 'What's currently open "
                         "and waiting on my review?'",
        "look_for": "An answer reflecting something actually true right now (live data), not a guess from training data.",
        "max_score": 5,
    },
    {
        "id": 14, "part": "Connectors, MCP & RAG",
        "title": "Write and Run a Five-Line MCP Tool",
        "prompt_given": "Write a FastMCP server exposing a search_orders(status) tool, run it, register it with "
                         "claude mcp add lab-tools -- python3 server.py, then call it from Claude Code.",
        "look_for": "A real MCP tool call appears in the transcript when asking Claude to 'search for pending orders using lab-tools'.",
        "max_score": 5,
    },
    {
        "id": 15, "part": "Connectors, MCP & RAG",
        "title": "Add One Hook That Blocks Something Real",
        "prompt_given": "Add a PreToolUse hook in .claude/settings.json that blocks any Bash command containing rm -rf, "
                         "restart, then try to run rm -rf /tmp/something.",
        "look_for": "The dangerous command was actually blocked before execution, with a custom message shown back — "
                    "enforced behavior, not just model discretion.",
        "max_score": 5,
    },
    {
        "id": 16, "part": "Connectors, MCP & RAG",
        "title": "Ground an Answer in Your Own Document",
        "prompt_given": 'Using only the text: "Employees accrue 1 paid sick day per month during their first year, up to 12 days." '
                         "Question: How many sick days in month 3?",
        "look_for": "The answer is grounded strictly in the pasted text (3 days) with no outside guessing, and explicitly "
                    "notes if something isn't covered by the text.",
        "max_score": 5,
    },
    {
        "id": 17, "part": "Bonus",
        "title": "Decide: One Path, or a Loop?",
        "prompt_given": "I need a workflow that grades retrieved documents and, if none are relevant, rewrites the query and "
                         "searches again. Straight pipeline or something that can loop back a step?",
        "look_for": "A recommendation that explicitly calls out looping/conditional branching (e.g. a graph-based "
                    "orchestrator) as the deciding factor, rather than a plain linear pipeline.",
        "max_score": 5,
    },
]

MAX_TOTAL_SCORE = sum(t["max_score"] for t in TASKS)

# ---------------------------------------------------------------------------
# GRADER (was api/grader.py)
# ---------------------------------------------------------------------------

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
    answers: Dict[str, str]
    results: Dict[str, Any]
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
    graph = get_graph()
    initial_state: GradingState = {
        "name": name, "email": email, "answers": answers, "results": {},
        "total_score": 0, "max_score": MAX_TOTAL_SCORE, "percentage": 0.0, "overall_feedback": "",
    }
    return graph.invoke(initial_state)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Claude Task Sheet — Submit &amp; Score</title>
<style>
  :root{
    --ink:#1b1f23;
    --paper:#eef0ec;
    --card:#ffffff;
    --rule:#d7dcd3;
    --accent:#2f5d50;
    --accent-soft:#e4ede9;
    --warn:#a6432f;
    --mono: "IBM Plex Mono", "SF Mono", Menlo, Consolas, monospace;
    --serif: "IBM Plex Serif", Georgia, serif;
    --sans: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  *{box-sizing:border-box;}
  body{
    margin:0;
    background:var(--paper);
    color:var(--ink);
    font-family:var(--sans);
    line-height:1.5;
  }
  header{
    padding:48px 24px 28px;
    max-width:880px;
    margin:0 auto;
  }
  .eyebrow{
    font-family:var(--mono);
    font-size:12px;
    letter-spacing:.14em;
    text-transform:uppercase;
    color:var(--accent);
    margin:0 0 10px;
  }
  h1{
    font-family:var(--serif);
    font-size:clamp(28px,4vw,42px);
    margin:0 0 12px;
    letter-spacing:-0.01em;
  }
  header p{
    max-width:60ch;
    color:#4a504a;
    margin:0;
  }
  main{
    max-width:880px;
    margin:0 auto;
    padding:0 24px 120px;
  }
  .identity{
    background:var(--card);
    border:1px solid var(--rule);
    border-radius:4px;
    padding:24px;
    margin-bottom:32px;
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:16px;
  }
  @media (max-width:560px){ .identity{grid-template-columns:1fr;} }
  .field label{
    display:block;
    font-family:var(--mono);
    font-size:11px;
    letter-spacing:.08em;
    text-transform:uppercase;
    color:#6b6f68;
    margin-bottom:6px;
  }
  .field input, .field textarea{
    width:100%;
    font-family:var(--sans);
    font-size:15px;
    padding:10px 12px;
    border:1px solid var(--rule);
    border-radius:3px;
    background:#fbfbf9;
    color:var(--ink);
  }
  .field input:focus, .field textarea:focus{
    outline:2px solid var(--accent);
    outline-offset:1px;
    background:#fff;
  }
  .part-divider{
    font-family:var(--mono);
    font-size:12px;
    letter-spacing:.12em;
    text-transform:uppercase;
    color:var(--accent);
    margin:44px 0 14px;
    padding-bottom:8px;
    border-bottom:1px solid var(--rule);
  }
  .task-card{
    background:var(--card);
    border:1px solid var(--rule);
    border-left:3px solid var(--accent);
    border-radius:4px;
    padding:20px 22px;
    margin-bottom:16px;
  }
  .task-num{
    font-family:var(--mono);
    font-size:12px;
    color:var(--accent);
  }
  .task-title{
    font-family:var(--serif);
    font-size:19px;
    margin:2px 0 10px;
  }
  .task-prompt{
    font-family:var(--mono);
    font-size:12.5px;
    color:#464b45;
    background:var(--accent-soft);
    border-radius:3px;
    padding:10px 12px;
    margin:0 0 14px;
    white-space:pre-wrap;
  }
  .task-card textarea{
    width:100%;
    min-height:110px;
    resize:vertical;
    font-family:var(--mono);
    font-size:13.5px;
    padding:12px;
    border:1px solid var(--rule);
    border-radius:3px;
    background:#fbfbf9;
  }
  .task-card textarea:focus{
    outline:2px solid var(--accent);
    outline-offset:1px;
    background:#fff;
  }
  .task-card textarea::placeholder{ color:#9aa096; }
  .score-badge{
    display:none;
    margin-top:10px;
    font-family:var(--mono);
    font-size:12.5px;
    padding:8px 10px;
    border-radius:3px;
    background:#f2f2ee;
    border:1px solid var(--rule);
  }
  .score-badge.show{ display:block; }
  .score-badge b{ color:var(--accent); }

  .submit-bar{
    position:sticky;
    bottom:0;
    background:linear-gradient(to top, var(--paper) 60%, transparent);
    padding:20px 0 0;
    margin-top:36px;
  }
  #submitBtn{
    width:100%;
    padding:16px;
    font-family:var(--sans);
    font-weight:600;
    font-size:16px;
    color:#fff;
    background:var(--accent);
    border:none;
    border-radius:4px;
    cursor:pointer;
  }
  #submitBtn:disabled{ background:#8a9791; cursor:not-allowed; }
  #submitBtn:hover:not(:disabled){ background:#264d42; }

  #statusLine{
    font-family:var(--mono);
    font-size:13px;
    text-align:center;
    margin-top:10px;
    min-height:18px;
    color:var(--warn);
  }
  #statusLine.ok{ color:var(--accent); }

  .result-panel{
    display:none;
    margin-top:28px;
    background:var(--card);
    border:1px solid var(--rule);
    border-radius:4px;
    padding:28px;
    text-align:center;
  }
  .result-panel.show{ display:block; }
  .result-score{
    font-family:var(--serif);
    font-size:52px;
    margin:6px 0;
    color:var(--accent);
  }
  .result-panel p{ color:#4a504a; margin:6px 0 0; }
</style>
</head>
<body>

<header>
  <p class="eyebrow">16 + 1 Tasks · Auto-graded</p>
  <h1>Claude &amp; Claude Code — Task Sheet</h1>
  <p>Do each task for real in Claude or Claude Code, then paste what actually happened into the box below it.
     Submit once at the bottom — your answers are graded automatically and your score is logged to the team sheet.</p>
</header>

<main>
  <form id="taskForm">
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

    <div id="tasksContainer">
      <p style="font-family:var(--mono); font-size:13px; color:#6b6f68;">Loading tasks…</p>
    </div>

    <div class="submit-bar">
      <button type="submit" id="submitBtn">Submit &amp; Get Scored</button>
      <div id="statusLine"></div>
    </div>
  </form>

  <div class="result-panel" id="resultPanel">
    <p class="eyebrow" style="margin:0;">Your Result</p>
    <div class="result-score" id="resultScore">–</div>
    <p id="resultVerdict"></p>
  </div>
</main>

<script>
  // Point this at your backend if it's not served from the same origin.
  const API_BASE = "";

  let TASKS = [];

  async function loadTasks(){
    const res = await fetch(`${API_BASE}/api/tasks`);
    TASKS = await res.json();
    renderTasks();
  }

  function renderTasks(){
    const container = document.getElementById('tasksContainer');
    container.innerHTML = '';
    let lastPart = null;

    TASKS.forEach(task => {
      if(task.part !== lastPart){
        const divider = document.createElement('div');
        divider.className = 'part-divider';
        divider.textContent = task.part;
        container.appendChild(divider);
        lastPart = task.part;
      }

      const card = document.createElement('div');
      card.className = 'task-card';
      card.innerHTML = `
        <div class="task-num">TASK ${task.id}</div>
        <div class="task-title">${task.title}</div>
        <div class="task-prompt">${escapeHtml(task.prompt_given)}</div>
        <textarea id="answer-${task.id}" placeholder="Paste what Claude actually gave you for this task…"></textarea>
        <div class="score-badge" id="badge-${task.id}"></div>
      `;
      container.appendChild(card);
    });
  }

  function escapeHtml(str){
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  document.getElementById('taskForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('submitBtn');
    const statusLine = document.getElementById('statusLine');
    const name = document.getElementById('empName').value.trim();
    const email = document.getElementById('empEmail').value.trim();

    if(!name || !email){
      statusLine.textContent = 'Please enter your name and email.';
      statusLine.className = '';
      return;
    }

    const answers = {};
    TASKS.forEach(task => {
      const el = document.getElementById(`answer-${task.id}`);
      answers[String(task.id)] = el ? el.value : '';
    });

    btn.disabled = true;
    btn.textContent = 'Grading…';
    statusLine.textContent = 'Sending to the grader — this can take a little while for 17 tasks.';
    statusLine.className = '';

    try{
      const res = await fetch(`${API_BASE}/api/submit`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name, email, answers})
      });
      const data = await res.json();

      if(!res.ok){
        statusLine.textContent = data.error || 'Something went wrong.';
        statusLine.className = '';
        btn.disabled = false;
        btn.textContent = 'Submit & Get Scored';
        return;
      }

      TASKS.forEach(task => {
        const r = data.results[String(task.id)];
        const badge = document.getElementById(`badge-${task.id}`);
        if(r && badge){
          badge.classList.add('show');
          badge.innerHTML = `<b>${r.score}/5</b> — ${escapeHtml(r.feedback)}`;
        }
      });

      document.getElementById('resultScore').textContent = `${data.total_score} / ${data.max_score}`;
      document.getElementById('resultVerdict').textContent =
        `${data.percentage}% — ${data.overall_feedback}` +
        (data.sheet_status === 'ok' ? ' (logged to the team sheet)' :
         data.sheet_status === 'not_configured' ? '' : ' (could not log to the sheet — ask your admin to check the webhook)');
      document.getElementById('resultPanel').classList.add('show');
      document.getElementById('resultPanel').scrollIntoView({behavior:'smooth'});

      statusLine.textContent = 'Done.';
      statusLine.className = 'ok';
      btn.textContent = 'Submitted';
    } catch(err){
      statusLine.textContent = 'Network error — is the backend running?';
      statusLine.className = '';
      btn.disabled = false;
      btn.textContent = 'Submit & Get Scored';
    }
  });

  loadTasks();
</script>

</body>
</html>
"""

# ---------------------------------------------------------------------------
# FLASK APP (was api/index.py)
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

SHEET_WEBHOOK_URL = os.environ.get("SHEET_WEBHOOK_URL", "").strip()


@app.route("/")
def serve_index():
    return app.response_class(INDEX_HTML, mimetype="text/html")


@app.route("/api/tasks")
def api_tasks():
    return jsonify(TASKS)


@app.route("/api/submit", methods=["POST"])
def api_submit():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    answers = data.get("answers") or {}

    if not name or not email:
        return jsonify({"error": "Name and email are required."}), 400
    if "@" not in email:
        return jsonify({"error": "Please enter a valid email."}), 400

    try:
        result = grade_submission(name, email, answers)
    except Exception as exc:  # noqa: BLE001
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
