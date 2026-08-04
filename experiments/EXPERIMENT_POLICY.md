# Experiment Policy — PROD is frozen

**PRODUCTION (do NOT edit during experiments):**
- `src/koscine3/largemove/` — the locked engine code
- `locks/prod_largemove_v1/` — locked config, universe, models, predictions, manifest, checksums
- `locks/prod_largemove_v2/` — **FROZEN** direction-agnostic mover engine (`mover_v2.py` + `universe_groups.json`)

These are the production large-move engines. They are **frozen**. No experiment may modify them.

**Verify integrity:**
- v1: `python experiments/clone_prod.py --verify`
- v2: `python experiments/freeze_v2_prod.py --verify`

The v2 **direction overlay** (`direction_stage2.py` / `direction_overlay.csv`) is a *separate, non-PROD* signal
under active research — it is NOT part of the frozen engine and may evolve.

## How to run an experiment

Every experiment must **clone PROD into its own sandbox** and work only on the copy:

```bash
python experiments/clone_prod.py <experiment_name>
```

This creates `experiments/<experiment_name>/` containing a **full self-contained copy** of:
- the `largemove/` package code,
- the locked `config`, `universe_groups.json`, and (optionally) `models/`.

Edit and run only inside `experiments/<experiment_name>/`. PROD is never imported or mutated.

```bash
cd experiments/<experiment_name>
python run_experiment.py            # uses the cloned copy, writes results into this folder
```

## Promotion

If an experiment beats PROD on the gold criteria (walk-forward precision per group, regime stability),
it is promoted by a **new lock** (`locks/prod_largemove_v2/`) and a new package version — never by editing v1.

## Integrity check

`python experiments/clone_prod.py --verify` recomputes checksums of `src/koscine3/largemove/` and
`locks/prod_largemove_v1/` against the locked ledger and fails if PROD was altered.
