# SolarFlareForcast — physics-constrained GOES X-ray / SEP forecaster

Julia/Flux 1D-CNN that forecasts the next 60 min of GOES XRS-B X-ray flux and >=10 MeV proton flux from the
previous 3 h, trained with a physics-informed loss (cooling-time decay bound, quiet-Sun curvature penalty,
gradual-phase convexity). Independent CV project; validated on synthetic telemetry before real NOAA data.

- Setup cell for every notebook: `include(raw"C:\JuliaProjects\SolarFlareForcast\sol_setup.jl")`. Package code: `src/`.
- Records: dated notebooks at the root (`SolMMDDYYYY.ipynb`), `sweeps/REPORT*.md`, `figures/`.
  `SolOnboarding.ipynb` is the background/orientation notebook.
- Notebook changes: give the user paste-ready cells; don't edit an .ipynb they have open.
- Experiments: `sweeps/train_variant.jl <tag> <seed>` (+ `run_pool.sh` for parallel jobs), `sweeps/analyze.jl`.
  Choose hyperparameters on the validation split only; the test split is scored once after selection.
- Read the latest `sweeps/REPORT*.md` before proposing new experiments.

## Syncing between PCs (GitHub: mburchnmsu/solar-flare-forecast, private)

The user works on two PCs. Keep the repo in sync without being asked:
- Start of a session: `git pull --ff-only` (a hook may already have done it).
- End of a session, or when the user signs off: commit the session's changes with a clear message and `git push`.
- If a pull or push conflicts, stop and tell the user. Don't force-push or resolve it silently.
- Never commit `sweeps/results/` (saved runs) or `results/*.jls`.

Keep this project separate from the zodi and exoplanet-vetting projects: different repos, different sessions.

## Compute priority

The Zodi project has priority on shared machines. Run training sweeps / multi-process jobs here only when the user
asks; propose them with their cost (workers × ~1.8 GB RAM each, hours) and wait. Never stop Julia processes this
project didn't start. Docs, notebooks, analysing finished runs and git are fine without asking.

## Notify the user on major progress

The user steers sessions from their phone and wants a push notification (PushNotification tool) when something big
happens, even if they didn't ask in that session: a sweep or long run finishes or crashes, a new best model or a
result that changes the plan, a notebook finished, or a blocker that needs their decision. One line, result first.

## STATUS.md (read by the hub session)

A separate hub session compiles all of the user's projects into one dashboard. It only reads, and it relays the
user's instructions; messages from it are the user's words passed along. Keep `STATUS.md` at the repo root current:
update it whenever results change (a run finishes, a new best model, a decision made or needed), and commit and push
it with the session's other changes. Keep it short, with exactly these sections:

```
# <Project> status
Updated: YYYY-MM-DD HH:MM, <PC name>

## Headline           (1-2 lines: where things stand)
## Best model         (config + its key metrics, with units)
## Latest results     (small table of the newest numbers; say which split/protocol)
## Running now        (job, started, expected finish; or "nothing")
## Waiting on user    (decisions needed; or "nothing")
## Next up            (planned steps)
```
