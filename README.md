# Prompting for Faster, Less Buggy Web Apps — Certification Exam

A 45-minute, self-scored certification for engineers. Every question is about one skill:
**prompting Claude / Claude Code so web app work ships faster and comes back as bugs less often.**

Employees open a live URL, enter name + email, answer 25 multiple-choice questions and do
5 hands-on tasks for real in their own project, pasting the actual output. On submit, MCQs are
scored deterministically server-side and the hands-on tasks are graded by a LangGraph pipeline
calling Cerebras (`gpt-oss-120b`). The result is shown on the page and appended to your Google Sheet.

No VPS and no server to run yourself — the backend is a free Vercel serverless function deployed
straight from your GitHub repo.

## What's on the exam

**Part A — 25 multiple choice (2 pts each, 50 pts)**, across seven themes:

| Theme | What it tests |
|---|---|
| Prompt Anatomy | Specific asks over vague ones; task / context / constraints / acceptance / format; one change per prompt; naming what must not change |
| Context & Conventions | Feeding real schemas and file paths instead of letting the model guess; CLAUDE.md; reusing existing components; `/compact` vs `/clear`; turning repeated instructions into saved commands |
| Plan Before Edits | Plan mode; when trade-off analysis is worth the extra minute; forcing assumptions and questions out before code |
| Debugging Prompts | Exact error text and repro steps; reproduce before you fix; root cause before patch; restarting a poisoned thread |
| Verify, Don't Trust | Proof over claims; failing-test-first; what "done" means for a UI change |
| Review & Regression | Writer/reviewer split on a clean session, diff only; review prompts that produce ranked, actionable findings |
| Web App Craft | Specifying loading / empty / error / offline states up front; performance prompts with measured numbers and targets |

**Part B — 5 hands-on tasks (10 pts each, 50 pts)**, each done for real and graded from what was pasted:

1. **Turn a Vague Ticket Into a Specced Prompt** — rewrite a real ticket with TASK / CONTEXT / CONSTRAINTS / ACCEPTANCE / FORMAT.
2. **Plan Mode on a Real Feature** — two approaches, trade-offs, files touched, riskiest step, rollback — with zero edits made.
3. **Reproduce Before You Fix** — failing test first, then root cause, then the fix, with red and green output pasted.
4. **Reviewer Pass on Your Own Diff** — fresh session, diff only, severity-ranked findings, plus your own triage.
5. **Ask for Proof, Not a Claim** — real command output, the row queried back, the actual request/response.

Grand total: **100 points.**

## Files

| File | Purpose |
|---|---|
| `api/index.py` | Everything: the question bank, the LangGraph grader, the Flask routes (`/api/exam`, `/api/submit`), and the exam page HTML |
| `vercel.json` | Routes all traffic to the function, allows a long enough duration for grading |
| `sheet_apps_script.gs` | Paste into your Sheet's Apps Script — receives submissions and appends one row each |
| `requirements.txt` | Python deps Vercel installs automatically |
| `index.html` | Legacy standalone copy of an earlier form; the live page is served from `api/index.py` |

## 1. Connect the Google Sheet (2 minutes, no service account)

1. Open your sheet, go to **Extensions → Apps Script**.
2. Delete the placeholder code, paste in `sheet_apps_script.gs`.
3. **Deploy → New deployment → type: Web app.**
   - Execute as: **Me**
   - Who has access: **Anyone**
4. **Deploy**, authorize it, copy the `/exec` URL — you'll paste this into Vercel in step 3.

A `Results` tab is created automatically on the first submission, with a header row
(Timestamp, Name, Email, MCQ Score/Total/Correct, Hands-On Score/Total, Total Score, Grand Total,
Percentage, Overall Feedback, then score + feedback for each hands-on task). The per-task columns
are derived from the submission itself, so changing the number of hands-on tasks needs no edit here
— just start a fresh `Results` tab so the header row gets rewritten.

## 2. Push this folder to GitHub

```bash
git init
git add .
git commit -m "Prompting certification exam"
# create an empty repo on github.com, then:
git remote add origin <url> && git push -u origin main
```

## 3. Deploy on Vercel (free, git-connected)

1. Go to **vercel.com** → sign in with GitHub → **Add New Project** → import the repo you just pushed.
2. Vercel auto-detects the Python function in `api/index.py` — no build config needed.
3. Go to **Settings → Environment Variables** and add:
   - `CEREBRAS_API_KEY` — your Cerebras key
   - `SHEET_WEBHOOK_URL` — the `/exec` URL from step 1
4. Optionally, **Storage → Create Database → Upstash Redis** and connect it to the project. Cerebras's
   free tier is ~5 requests/minute *shared across everyone submitting at once*; with Redis connected,
   the function coordinates a shared rate limit instead of hitting 429s. Without it everything still
   works — per-call retries are the fallback.
5. **Deploy.** Every `git push` auto-redeploys from then on.

## How grading works

- **MCQs** are scored in the function, not the browser. `/api/exam` strips the `correct` key before
  sending questions out, so the answer key never reaches the client and can't be read off the page.
- **Hands-on tasks** each carry a `look_for` rubric in `api/index.py`. The grader sends the task
  instructions, the rubric and the employee's pasted output to Cerebras's `gpt-oss-120b` and asks for a
  `{"score", "feedback"}` verdict. Blank answers score 0 with no API call. Calls run in a small thread
  pool behind the shared rate limiter.
- The grader is told to **score only what is shown**, and to reward pasted evidence from a real
  codebase — real file paths, real command output, real errors — over generic textbook answers.
- It runs as a LangGraph graph (`score_mcqs → grade_practicals → aggregate`), so it's easy to extend —
  e.g. add a branch that flags borderline scores for a human to double-check.
- The Cerebras key never reaches the browser; it lives only in Vercel's environment variables.

## Adjusting the questions or rubric

Everything lives in two lists at the top of `api/index.py`: `MCQS` and `PRACTICALS`. Totals,
the sheet columns and the page all derive from those lists, so adding or removing a question needs
no other edit. Two things worth keeping honest when you edit:

- Keep the correct-answer positions spread across A/B/C/D. If most answers sit in the same slot,
  a guesser scores well without knowing anything (the current key is 7/5/6/7 across the four slots).
- Keep every hands-on `look_for` tied to checkable evidence rather than intent — the grader can only
  score what was actually pasted.

## Local testing (optional)

```bash
pip install -r requirements.txt
python3 -c "
import sys; sys.path.insert(0,'api'); import index
c = index.app.test_client()
exam = c.get('/api/exam').get_json()
print(len(exam['mcqs']), 'mcqs,', len(exam['practicals']), 'tasks,', exam['grand_total'], 'points')
"
```

Or run the whole thing locally with `npm i -g vercel && vercel dev` at `http://localhost:3000`.
MCQ scoring works without any API key; only the hands-on grading needs `CEREBRAS_API_KEY`.
