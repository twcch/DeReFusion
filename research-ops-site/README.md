# Research OS prototype

Local, interactive research-operations dashboard for the DeReFusion project.

## Run locally

```powershell
npm ci
npm run dev
```

Open `http://localhost:5173/`.

## Included views

- Project overview and stage gate
- Experiment registry with status filtering
- Frozen protocol and Stage B gate checks
- Claims and evidence ledger
- Agent task board with duplicate-work lock
- Local/cloud compute and budget view
- Light/dark theme and responsive mobile navigation

The displayed Stage A metrics are explicitly labelled as execution calibration only. This prototype does not start experiments, purchase compute, mutate the repository, or persist UI changes.

## Validation

```powershell
npm run build
```

The prototype uses the repository's current DeReFusion research status as sample data. It is ready to be connected to manifests, `receipt.json` artifacts, Git status, and a persistent database in a later iteration.
