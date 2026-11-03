# Data contract

This lets Kiet and Nhan work in parallel: TabPFN results arrive in the same format as classic results, so analysis and
interface code never need to know where a grid came from. The source of truth is `src/tabbias/contract.py`.

## Run key
`dataset / variant / n / seed / model`, for example `spirals_medium/clean/n60/seed0/rf`.

| Field | Values |
|---|---|
| dataset | `moons circles xor spirals_loose spirals_medium spirals_tight stripes checker_fine`; real datasets added later |
| variant | `clean rot45 noise10 noise20` |
| n | `10 30 60 200` (nested contexts) |
| seed | `0`–`9` |
| model | classic: `logreg knn rf gbdt svm mlp gp`; TabPFN: `tabpfn_v2 tabpfn_v35`, with `_e1 _e4 _e8` for ablation and `tabpfn_v35_fast` as a fallback |

## Files
| Path | Contents | In Git? |
|---|---|---|
| `results/grids/<key>.npz` | `p` (200×200 float32, P(class 1)), `half_width`, `contract_version` | No: shared Drive, then a GitHub Release |
| `results/runs/<key>.json` | One run's metrics; written last, so its existence means the run finished | No |
| `results/metrics/*.csv` | Aggregated metrics, one row per run | **Yes** |
| `interface/data/fixture/` | Small browser sample for development | **Yes** |

Grids are square, span `[-half_width, half_width]` on both axes in standardized units, with rows following y and columns following x, both ascending.

## Metrics columns
`dataset, variant, n, seed, model, accuracy, log_loss, ece, far_field_confidence, jaggedness, axis_alignment, jump_share, transition_width, fit_seconds, predict_seconds, device`

## Browser format
`scripts/export_interface.py` writes `index.json` and one JSON file per grid:
```json
{"dataset": "spirals_medium", "variant": "clean", "n": 60, "seed": 0, "model": "rf",
 "res": 100, "half_width": 3.1, "p_uint8_b64": "...", "metrics": {"accuracy": 0.9, "jump_share": 0.6}}
```
`p_uint8_b64` is `res × res` base64-encoded bytes in row order, where probability = byte / 255.

The fixture does not include the context points yet. Add them to the export (Issue for the slider) so the canvas can draw them.
