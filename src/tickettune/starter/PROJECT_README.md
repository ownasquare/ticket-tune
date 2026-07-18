# My TicketTune project

This starter fine-tunes a small instruction model for structured support-ticket triage.
Its sample data is synthetic and contains visible placeholders instead of personal data.

## Confirm the CLI

```bash
tickettune --version
```

If that command is unavailable, return to the TicketTune source checkout and run
`uv tool install .` once. The installed command then works from this directory.

## Prove the project without downloading a model

```bash
tickettune data validate --config configs/tickettune.yaml
tickettune data prepare --config configs/tickettune.yaml
tickettune train --config configs/tickettune.yaml --dry-run
tickettune evaluate --config configs/tickettune.yaml --predictions predictions/pass.jsonl --enforce-thresholds
```

## Fine-tune when you are ready

From the TicketTune source checkout, add the training dependencies once with
`uv tool install --force '.[train]'`. Then return here and run:

```bash
tickettune train --config configs/tickettune.yaml --allow-download
```

The starter uses a one-step teaching profile. Read the main TicketTune getting-started and
customization guides before increasing model size, changing the dataset, or deploying a model.
