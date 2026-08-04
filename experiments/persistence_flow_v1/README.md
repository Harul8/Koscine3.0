# Experiment: persistence_flow_v1

Self-contained clone of PROD (`prod_largemove_v1`). Edit `largemove/` here — PROD is frozen.

```
python run_experiment.py
```

Artifacts (predictions/, models/) are written under this folder. When done, compare against PROD; promote only via a new lock (see EXPERIMENT_POLICY.md).
