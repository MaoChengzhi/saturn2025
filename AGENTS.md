# Saturn2025

This repository supports Cassini MAG and Juno MAG research on ion cyclotron wave (ICW) and mirror-like candidates. Work in the original local checkout unless I request a worktree. I inspect changes in VS Code and run selected cells in Jupyter.

## Work with me

- Before editing, inspect the current Git status and the relevant source. When I ask you to make a change, proceed within that scope without asking again. Ask before expanding the scope, and preserve my existing edits.
- Keep each repair small enough to review. Reuse existing code and avoid reorganizing notebooks or adding files unless the task calls for it. Preserve cell order, comments, and saved outputs unless we agree otherwise.
- Treat raw data and kernels as read-only. Ask before installing packages, running a full archive job, committing, or pushing.
- After editing, report which files and cells changed and what you verified. Show the **View changes** card when the app provides it. If the card is absent, show or link the diff for this task against the state immediately before the edit. For notebooks, show changed cell source rather than raw notebook JSON. If only a combined diff is available, say so.

## Research checks

- Use the existing `saturn2025` Python environment. On this Mac the interpreter is `/opt/anaconda3/envs/saturn2025/bin/python`. State which notebook cells you ran and which ones I should run to reproduce the result.
- Start with the smallest relevant interval or a controlled signal. For path-only changes, use the December 21, 2010 FFT example as a regression check when relevant. Its historical reference is 600 processed samples at one-second cadence, mean field about 185.766 nT, and a perpendicular peak near 0.143333 Hz.
- Call out proposed changes to units, frames, background windows, sampling, PSD, polarization, or classification thresholds. State the assumptions and cite the relevant notes or paper when changing a scientific rule.
- Check timestamp gaps and usable frequency coverage before applying a uniform-grid FFT. Keep invalid or unobservable windows separate from eligible non-detections. For occurrence statistics, report coverage, candidate counts, eligible denominators, and the handling of windows flagged for both modes.
