# FlashTL-1K Dataset Tools

`extract_context.py` reproduces the optional contextual-analysis pipeline used to
estimate traffic-light localization, depth, temporal signal descriptors, and
scene attributes.

The generated `context_results.csv` can be supplied to `train.py` so the
temporal brightness trace is extracted from the detected traffic-light bounding
box when available.

The contextual pipeline has heavier dependencies than core FlashNet. Install:

```bash
pip install -r requirements-context.txt
```

SAM2 may require separate installation/checkpoints from its official project.
