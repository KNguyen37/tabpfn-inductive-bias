---
name: next-task
description: Find and start this teammate's next unblocked GitHub Issue for the TabPFN project. Use at the start of a work session or when the user asks what to do next.
---

# Next task

1. Run `git status`. If there are uncommitted changes, stop and ask the user what to do with them.
2. Run `git switch main && git pull --ff-only`.
3. Identify the teammate with `gh api user --jq .login`: `KNguyen37` is `owner:kiet`, `NhanDoan1203` is `owner:nhan`. For anyone else, stop and ask.
4. If `.venv` is missing, follow the setup in `CLAUDE.md` and run the tests.
5. List tasks with `gh issue list --state open --label owner:<name> --json number,title,labels,body`.
6. An Issue is blocked if its body has `Blocked by #N` and #N is still open. Choose the first unblocked Issue, preferring `priority:today`, then the lowest number.
7. Show the user the task and its acceptance criteria in 2–3 lines. If it has `compute:colab`, and the user is not Kiet, stop: leave a comment on the Issue for Kiet.
8. Once the user confirms, create a branch named `<name>/<issue>-<slug>` and hand the work to the agent named in the Issue's `agent:` line.
9. When finished: run the tests, run the `auditor` agent, push, and open a PR with `Closes #N`. Do not merge it. Tell the user the PR link.

If a result changes a hypothesis, metric or decision rule, stop and ask the user instead of deciding.
