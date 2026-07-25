# Claude Task Sheet — Auto-Grader (live via GitHub + Vercel, no server to manage)

Employees open a live URL, enter name + email, do the 17 tasks in Claude/Claude Code
for real, and paste output under each. On submit, a LangGraph pipeline scores every
answer with Cerebras (`gpt-oss-120b`), and the result is appended to your Google Sheet.

This version needs nothing but **git** and **the sheet** — no VPS, no server to run
yourself. The backend runs as a free Vercel serverless function, deployed straight
from your GitHub repo.

## Files

| File | Purpose |
|---|---|
| `index.html` | The employee-facing form — served as-is by Vercel |
| `api/index.py` | Flask app Vercel runs as a serverless function — `/api/tasks`, `/api/submit` |
| `api/tasks.py` | The 17 tasks + rubric (single source of truth for frontend + grader) |
| `api/grader.py` | LangGraph graph (`grade_all_tasks → aggregate`), calls Cerebras in parallel |
| `vercel.json` | Routes `/api/*` to the function, allows 60s for the 17 parallel grading calls |
| `sheet_apps_script.gs` | Paste into your Sheet's Apps Script — receives & appends rows |
| `requirements.txt` | Python deps Vercel installs automatically |

## 1. Connect the Google Sheet (2 minutes, no service account)

1. Open your sheet, go to **Extensions → Apps Script**.
2. Delete the placeholder code, paste in `sheet_apps_script.gs`.
3. **Deploy → New deployment → type: Web app.**
   - Execute as: **Me**
   - Who has access: **Anyone**
4. **Deploy**, authorize it, copy the `/exec` URL — you'll paste this into Vercel in step 3.

A `Results` tab is created automatically on the first submission, with a header row
(Timestamp, Name, Email, Total Score, Max Score, Percentage, Overall Feedback, then
per-task score + feedback for all 17 tasks).

## 2. Push this folder to GitHub

```bash
cd task-sheet-vercel
git init
git add .
git commit -m "Claude task sheet auto-grader"
gh repo create claude-task-sheet --private --source=. --push
# (or: create an empty repo on github.com, then `git remote add origin <url>` and `git push -u origin main`)
```

## 3. Deploy on Vercel (free, git-connected)

1. Go to **vercel.com** → sign in with GitHub → **Add New Project** → import the repo you just pushed.
2. Vercel auto-detects the Python function in `api/index.py` and the static `index.html` — no build config needed.
3. Before the first deploy (or right after, then redeploy), go to **Settings → Environment Variables** and add:
   - `CEREBRAS_API_KEY` — your Cerebras key
   - `SHEET_WEBHOOK_URL` — the `/exec` URL from step 1
4. **Deploy.** You get a live URL immediately (e.g. `https://claude-task-sheet.vercel.app`).

From here, every `git push` auto-redeploys — no manual steps, no server to keep running.

## How grading works

- Each task has a rubric line (`look_for` in `api/tasks.py`) worth 0–5 points, 17 tasks = 85 points max.
- On submit, `api/grader.py` sends each task's instructions + rubric + the employee's
  pasted text to Cerebras's `gpt-oss-120b`, in parallel (all 17 calls fire at once via
  a thread pool), and asks for a `{"score", "feedback"}` verdict. Blank answers score 0
  with no API call.
- This runs as a LangGraph graph (`grade_all_tasks -> aggregate`) so it's easy to extend —
  e.g. add a branch that flags borderline scores for a human to double-check.
- The Cerebras key never reaches the browser — it lives only in Vercel's environment
  variables and is used inside the serverless function.

## Adjusting the rubric or task text

Everything about the 17 tasks lives in one place: `api/tasks.py`. Edit it there — both
the frontend (`/api/tasks`) and the grader read from the same list.

## Local testing (optional)

```bash
npm i -g vercel
vercel dev
```
This runs the same setup locally at `http://localhost:3000`, reading `.env` for the
two variables (copy `.env.example` to `.env` first). Not required — you can also just
push straight to Vercel and test on the live URL.
