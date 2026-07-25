"""
The 17 tasks from the Claude & Claude Code Daily Productivity Task Sheet.
Each task carries the instruction + the "Look for" criteria the grader checks against.
"""

TASKS = [
    {
        "id": 1,
        "part": "Prompting & Planning",
        "title": "Turn a Vague Ask Into a Tagged Prompt",
        "prompt_given": 'Rewrite this using task, context, constraints, and format: "Can you add caching to the search endpoint, '
                         'it\'s kind of slow, don\'t break the existing tests"',
        "look_for": "A response with the request clearly split into task / context / constraints / format sections — "
                    "nothing left for Claude to guess at.",
        "max_score": 5,
    },
    {
        "id": 2,
        "part": "Prompting & Planning",
        "title": "Fill a 5-Line Spec Block Before You Build",
        "prompt_given": "Write a TASK / DATA / SERVICES / CONSTRAINTS / FORMAT block for: when a user cancels a subscription, "
                         "write a row to a cancellations table and call the billing provider's refund API for partial-month refunds.",
        "look_for": "Five short labeled lines (TASK, DATA, SERVICES, CONSTRAINTS, FORMAT), each substantive and specific "
                    "to the cancellation/refund scenario.",
        "max_score": 5,
    },
    {
        "id": 3,
        "part": "Prompting & Planning",
        "title": "Force Plan Mode Before Anything Gets Edited",
        "prompt_given": "Don't write any code yet. First give me a plan: files touched, approach, and risks. I'll approve it, "
                         "then you implement. (or Shift+Tab into Plan Mode)",
        "look_for": "Evidence Claude proposed a plan (files touched, approach, risks) and paused for approval before editing anything.",
        "max_score": 5,
    },
    {
        "id": 4,
        "part": "Prompting & Planning",
        "title": "Start Minimal, Then Correct Instead of Front-Loading",
        "prompt_given": "I need to debug why our token refresh sometimes fails silently. Start by asking me whatever you need "
                         "to know — don't assume the stack yet.",
        "look_for": "Claude asked a short, targeted clarifying question back instead of guessing the architecture.",
        "max_score": 5,
    },
    {
        "id": 5,
        "part": "Context & Cost Management",
        "title": "Generate Your Memory File, Then Trim It",
        "prompt_given": "/init",
        "look_for": "A CLAUDE.md was generated reflecting the real project stack/conventions, and the submitter describes "
                    "trimming anything not actively true.",
        "max_score": 5,
    },
    {
        "id": 6,
        "part": "Context & Cost Management",
        "title": "Use /compact and /clear for What They're Actually For",
        "prompt_given": "/compact (mid-task, same ticket) and /clear (switching to an unrelated task)",
        "look_for": "Evidence /compact shrank context while keeping task memory, and /clear was used specifically when "
                    "switching to unrelated work.",
        "max_score": 5,
    },
    {
        "id": 7,
        "part": "Context & Cost Management",
        "title": "Check What a Session Actually Costs",
        "prompt_given": "/cost",
        "look_for": "A real cost/usage breakdown for the session was retrieved and reported.",
        "max_score": 5,
    },
    {
        "id": 8,
        "part": "Context & Cost Management",
        "title": "Run a Second Task in Its Own Worktree",
        "prompt_given": "git worktree add ../my-repo-hotfix -b hotfix/quick-fix && cd ../my-repo-hotfix && claude",
        "look_for": "A second, isolated Claude Code session running on its own branch/worktree, separate from the main session.",
        "max_score": 5,
    },
    {
        "id": 9,
        "part": "Verification & Delegation",
        "title": "Ask for Proof, Not a Claim",
        "prompt_given": "When done, show me: 1) actual test output, 2) the row it wrote, queried back for real, 3) any step "
                         "that failed silently.",
        "look_for": "Real, queryable output/evidence was returned — not just an assertion that something works.",
        "max_score": 5,
    },
    {
        "id": 10,
        "part": "Verification & Delegation",
        "title": "Get a Genuinely Fresh Review",
        "prompt_given": "/clear, then paste only the diff: Review this diff for security, edge cases, and readability. "
                         "Report issues — don't rewrite it.",
        "look_for": "A fresh-context review (post /clear) surfaced issues the original session had missed, without rewriting the code.",
        "max_score": 5,
    },
    {
        "id": 11,
        "part": "Verification & Delegation",
        "title": "Delegate Research to a Subagent",
        "prompt_given": "Use a subagent to investigate how our authentication handles token refresh, and whether we already "
                         "have an OAuth utility to reuse. Report back a summary, not the raw files.",
        "look_for": "A concise summary came back from a subagent, with the main session's context kept clean of raw file contents.",
        "max_score": 5,
    },
    {
        "id": 12,
        "part": "Verification & Delegation",
        "title": "Turn Your Most Repeated Prompt Into a Command",
        "prompt_given": "Create .claude/commands/int-test.md with a description and body that writes a real integration test "
                         "for $ARGUMENTS hitting the endpoint and confirming the DB row.",
        "look_for": "A working slash command file was created, and invoking /int-test <arg> runs the full instruction with zero retyping.",
        "max_score": 5,
    },
    {
        "id": 13,
        "part": "Connectors, MCP & RAG",
        "title": "Turn On One Real Connector",
        "prompt_given": "Settings → Connectors → Connect (GitHub or Google Drive) → approve. Then ask: 'What's currently open "
                         "and waiting on my review?'",
        "look_for": "An answer reflecting something actually true right now (live data), not a guess from training data.",
        "max_score": 5,
    },
    {
        "id": 14,
        "part": "Connectors, MCP & RAG",
        "title": "Write and Run a Five-Line MCP Tool",
        "prompt_given": "Write a FastMCP server exposing a search_orders(status) tool, run it, register it with "
                         "claude mcp add lab-tools -- python3 server.py, then call it from Claude Code.",
        "look_for": "A real MCP tool call appears in the transcript when asking Claude to 'search for pending orders using lab-tools'.",
        "max_score": 5,
    },
    {
        "id": 15,
        "part": "Connectors, MCP & RAG",
        "title": "Add One Hook That Blocks Something Real",
        "prompt_given": "Add a PreToolUse hook in .claude/settings.json that blocks any Bash command containing rm -rf, "
                         "restart, then try to run rm -rf /tmp/something.",
        "look_for": "The dangerous command was actually blocked before execution, with a custom message shown back — "
                    "enforced behavior, not just model discretion.",
        "max_score": 5,
    },
    {
        "id": 16,
        "part": "Connectors, MCP & RAG",
        "title": "Ground an Answer in Your Own Document",
        "prompt_given": 'Using only the text: "Employees accrue 1 paid sick day per month during their first year, up to 12 days." '
                         "Question: How many sick days in month 3?",
        "look_for": "The answer is grounded strictly in the pasted text (3 days) with no outside guessing, and explicitly "
                    "notes if something isn't covered by the text.",
        "max_score": 5,
    },
    {
        "id": 17,
        "part": "Bonus",
        "title": "Decide: One Path, or a Loop?",
        "prompt_given": "I need a workflow that grades retrieved documents and, if none are relevant, rewrites the query and "
                         "searches again. Straight pipeline or something that can loop back a step?",
        "look_for": "A recommendation that explicitly calls out looping/conditional branching (e.g. a graph-based "
                    "orchestrator) as the deciding factor, rather than a plain linear pipeline.",
        "max_score": 5,
    },
]

TASK_BY_ID = {t["id"]: t for t in TASKS}
MAX_TOTAL_SCORE = sum(t["max_score"] for t in TASKS)
