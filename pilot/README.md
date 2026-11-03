# Pilot code from the proposal

Kept unmodified as a record of how the proposal's numbers were produced. New work belongs in `src/tabbias/`.

Do not build on it. Known issues, all addressed in the new code:
- Model constructors fix `random_state=0`, so "seeds" varied the data but not the model.
- Outputs are hardcoded to `/home/claude/`.
- `axis.py` computes the proposal's step score S, which measures gradient direction, not sharpness.
- J is multiplied by 100 in `prelim.py`, but Eq. 1 in the proposal does not show this.
