# TabPFN inductive bias: instructions for Claude Code

Course project for DS-GA 3001 (Visualization for Machine Learning) by Kiet Nguyen and Nhan Doan.
Final deliverable: an 8-page IEEE VIS paper, this repo, and a D3 interface, due Dec 12, 2026.

Read these when relevant:
- `docs/PROJECT.md`: research questions and hypotheses
- `docs/DECISIONS.md`: decisions already made; do not reopen them without a teammate's say
- `docs/DATA_CONTRACT.md`: run keys, grid format, metrics columns, browser format
- `docs/PLAYBOOK.md`: how the team, machines and agents work together

## Who is running this session
Run `gh api user --jq .login`: `KNguyen37` is Kiet and `NhanDoan1203` is Nhan.
Use `/next-task` to find what to do. Tasks are GitHub Issues labeled `owner:kiet` or `owner:nhan`.

## Non-negotiable rules
1. **TabPFN runs only on Kiet's Colab GPU.** Never install or run it on a laptop. On Nhan's machine,
   hand any `compute:colab` work to Kiet by leaving a comment on the Issue.
2. **The interface is D3 only.** No Plotly, Streamlit, Dash or similar. Matplotlib is fine for static paper figures.
3. **No controlled user studies.** No timed tasks or accuracy comparisons across conditions.
   Formative walkthroughs reported qualitatively are allowed.
4. **Every reported number must trace to a committed file** in `results/`. Never write a result
   that has not been produced. If an experiment failed, report the failure.
5. **Do not change a hypothesis, metric definition or decision rule after seeing TabPFN results**
   unless a human approves it and the change is logged in `docs/DECISIONS.md` with its reason.
6. **Never commit secrets** (`TABPFN_TOKEN`) or large outputs (`results/grids`, `results/runs`).
7. **Humans merge.** Agents work on a branch and open a PR. `main` is protected.

## Setup
```bash
python3.11 -m venv .venv            # any Python >= 3.11
.venv/bin/pip install -r requirements-local.txt && .venv/bin/pip install -e . --no-deps
.venv/bin/python -m pytest -q
```

## Commands
- Tests: `.venv/bin/python -m pytest -q`
- Classic sweep, resumable: `.venv/bin/python scripts/run_sweep.py --family classic`
- Metric check on known surfaces: `.venv/bin/python scripts/metric_sanity.py`
- Browser data: `.venv/bin/python scripts/export_interface.py --dest interface/data/fixture ...`
- Interface: `python3 -m http.server -d interface 8000`

## Layout
`src/tabbias/` library · `scripts/` entry points · `tests/` · `notebooks/` Colab · `results/metrics/` committed CSVs ·
`interface/` D3 page · `paper/` IEEE VIS paper · `update/` project update · `pilot/` proposal code, kept unmodified

## Working style
- One Issue, one branch (`kiet/<issue>-slug` or `nhan/<issue>-slug`), one PR. Include `Closes #N`.
- Stay in your task's paths. Ask before editing `src/tabbias/contract.py`, because both teammates depend on it.
- Add a test for each new metric, and check it on data of known shape before using it.
