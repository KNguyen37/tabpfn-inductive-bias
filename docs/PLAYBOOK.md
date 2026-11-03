# Team and agent playbook

## How coordination works
Claude Code sessions on different machines cannot see each other and share no memory. **The repo is the only shared
channel:** `CLAUDE.md` and `.claude/` give both machines the same rules and agents, Issues are the task board, and the data contract keeps work independent.

```text
GitHub: CLAUDE.md, agents, Issues, data contract, metrics CSVs, PRs
   | pull/PR                | pull/PR                  | clone + run
Kiet's laptop agents     Nhan's laptop agents       Colab notebook (no agent)
   +-- download results -- shared Drive folder <---- grids + run records
```

## Who does what
| | Kiet | Nhan |
|---|---|---|
| **Compute** | Colab GPU, plus laptop | Laptop |
| **Owns** | TabPFN runs, metrics, hypothesis analysis, paper integration | Classic sweeps, real datasets, D3 interface |
| **Shared** | Update, paper writing, walkthroughs, reviewing each other's PRs | Same |

## Daily loop
1. Run `/next-task` in Claude Code. It pulls `main`, identifies you, and picks your next unblocked Issue.
2. The session works on a branch, using the matching agent, and opens a PR that includes `Closes #N`.
3. The auditor checks the PR. A human merges it: on GitHub, or by typing `! gh pr merge <N> --squash` in Claude Code.

Agents never merge, force-push, or change a hypothesis, metric definition or decision rule on their own.

## Agents
| Agent | Runs on | Job |
|---|---|---|
| `local-runner` | Nhan | Classic sweeps, dataset screening |
| `colab-bridge` | Kiet | Prepare notebooks and job lists; verify and import Colab results |
| `metrics-analyst` | Kiet | Metrics, sanity tests, hypothesis decisions |
| `d3-builder` | Nhan | D3 interface |
| `paper-writer` | Both | Update and paper, with every number traced to a file |
| `auditor` | Both | Read-only review of PRs against the rules |

Use one agent per task. Run several at once only when their files do not overlap.

## Colab workflow
Colab runs a notebook, not an agent, because sessions can disconnect at any time. Every finished run is saved to Drive and skipped on restart.

1. **One time:** accept the license at [ux.priorlabs.ai](https://ux.priorlabs.ai), create a token, and store it as `TABPFN_TOKEN` in Colab Secrets. Never paste it into a notebook cell.
2. Run `notebooks/colab_smoke_test.ipynb`, then commit the resulting `results/smoke/tabpfn_smoke.json`.
3. Run `notebooks/colab_sweep.ipynb`, rerunning it after any disconnect.
4. Use `colab-bridge` to verify the CSV and open a PR.

## Timeline
| Dates | Kiet | Nhan |
|---|---|---|
| **Nov 3** | Repo, model access test, metric check, update | Setup, classic sweep, dataset screen, D3 canvas (stretch), update review |
| **Nov 4–10** | Freeze metrics, hypotheses and margins | Finish classic sweep; canvas and controls |
| **Nov 11–17** | TabPFN synthetic sweep; RQ1, RQ3 analysis | Small multiples, metrics panel |
| **Nov 18–24** | TabPFN real-data runs; RQ2 analysis | Slider and compare mode; real-data baselines |
| **Nov 25–Dec 1** | Full paper draft | Story mode; both run walkthroughs |
| **Dec 2–8** | Results freeze Dec 5; paper revisions | Clean-clone README test; polish |
| **Dec 9–12** | Submit by Dec 11 | Dec 12 is buffer |

## Course requirements checklist
- [ ] 8-page IEEE VIS (VGTC/TVCG) paper with author names; references excluded from the page count
- [ ] Paper covers motivation and contributions, related work, methods, experiments, results and discussion
- [ ] Repo has source, data or a download script, and a README that reproduces the figures
- [ ] Interfaces built only in D3
- [ ] No controlled user studies
- [ ] Paper stands on its own without supplementary material
