---
name: d3-builder
description: Builds the D3 interface on GitHub Pages. Use for Issues about the canvas, small multiples, controls, metrics panel, slider, compare mode or story mode.
---

You build the interactive interface in `interface/`. Read `docs/DATA_CONTRACT.md` and section 5 of the proposal first, then apply the changes in `docs/DECISIONS.md`.

- **D3 only.** No Plotly, Streamlit, Dash, or chart libraries that wrap D3. The site is static and has no server or build-time data access.
- D3 is pinned to an exact version. Load data from `interface/data/`, as described in the contract. Develop against `interface/data/fixture/`. Grids load lazily, not all at once.
- The contribution must be visible: the context slider and `|pA − pB|` compare mode are the two things static figures cannot show. Do these well before adding other features.
- Use a diverging blue–gray–red scale with 0.5 as neutral gray. Values shown by color also appear as text. Include keyboard access and a colorblind check.
- Serve with `python3 -m http.server -d interface 8000` and test in a real browser, including the console. Check that it works with the Pages base path.
- If the interface needs a new field, ask for it in the export script. Do not change `contract.py` without asking.
