# Dataset qualification

TicketTune keeps dataset preparation, human review, model evaluation, and deployment as separate
proof layers. The checked-in 56-row teaching corpus is useful for smoke tests; it is not release
data. The release-oriented profile uses a deterministic 1,120-row synthetic candidate and does not
become qualified until two real people independently approve every row.

## The complete workflow

Create the candidate and its frozen train, validation, and test split:

```bash
uv run tickettune data generate-candidate \
  --output data/qualified/support_tickets.jsonl \
  --seed 42
uv run tickettune data prepare --config configs/qwen-7b-qlora-quality.yaml
```

Create a new private review workspace:

```bash
uv run tickettune qualify scaffold-review \
  --config configs/qwen-7b-qlora-quality.yaml \
  --output-dir data/qualified/review-evidence
```

The workspace contains:

```text
data/qualified/review-evidence/
├── prepared-manifest.json   # exact prepared-data identity
├── holdout-freeze.json      # exact ordered test IDs
├── reviewer-a.json          # first human's 1,120 decisions
├── reviewer-b.json          # second human's 1,120 decisions
└── review-manifest.json     # aggregate v1.2 attestation
```

The scaffold is deliberately non-approving: both reviewer IDs are placeholders, every decision is
`pending`, dates are empty, and statuses are `draft`. Generated files are evidence templates, not
human review. Do not hand-copy the checked-in example manifest; the scaffold command creates the
exact file references and hashes for this candidate.

Give `reviewer-a.json` and `reviewer-b.json` to two different people. Each reviewer works
independently and must:

1. replace the placeholder with their own distinct, non-placeholder reviewer ID;
2. inspect every row in source order;
3. decide `labels`, `response`, `pii`, and `license` for every row;
4. set an actual `review_date`; and
5. set packet `status` to `approved` only if all 1,120 decisions are `approved`.

If either person rejects anything, keep the packet or aggregate rejected. Fixing source bytes
invalidates the prepared hashes and requires preparation and both reviews to start again. Do not
change the frozen test cohort after training or tuning begins.

After both packets are complete, set the aggregate `review_date` and `approval_status` honestly,
then refresh its file hashes without overwriting the edited input:

```bash
uv run tickettune qualify bind-review \
  --review-manifest data/qualified/review-evidence/review-manifest.json \
  --output data/qualified/review-evidence/review-manifest.bound.json
```

`bind-review` only records the edited files' hashes. It never approves a decision. Enforce the full
gate before training:

```bash
uv run tickettune qualify dataset \
  --config configs/qwen-7b-qlora-quality.yaml \
  --review-manifest data/qualified/review-evidence/review-manifest.bound.json \
  --output artifacts/qualified-candidate/qualification-report.json \
  --enforce
```

The quality profile points to that bound aggregate, so a real training run rechecks the same
evidence before importing the optional ML stack or allocating a run directory.

## What the v1.2 gate checks

Qualification fails closed unless all of these are true:

- the source has at least 1,000 valid records and exactly matches the aggregate hash and count;
- the prepared manifest partitions every source ID exactly once;
- the frozen holdout exactly matches the ordered prepared test split and contains at least 100 IDs;
- two distinct packet files identify two distinct, non-placeholder human reviewers;
- both packets bind the exact source, prepared manifest, and holdout freeze;
- each packet covers every source ID in source order;
- every label, response, PII, and license decision is `approved` in both packets;
- both packets have real dates and approved status; and
- the aggregate has a review date, approved status, synthetic-only origin, and the required license,
  PII, domain, and holdout attestations.

The legacy v1.1 count-only manifest remains readable for diagnostics, but is always
`qualified: false`. Integer reviewer counts cannot prove independent review.

## Privacy and repository safety

`data/qualified/` is ignored except for the deliberately unapproved
`review-manifest.example.json`. Keep the candidate and review workspace out of Git. Use only
synthetic records; never copy real customer tickets into this workflow.

Automated audits are useful supplemental checks, but they do not count as either human reviewer,
cannot edit or approve packets, and cannot make the dataset release-eligible.

## Proof boundary

A passing qualification report proves that the exact synthetic bytes meet the packet-backed review
policy. It does not prove that the data represents production traffic, that a trained model meets
quality thresholds, that adapter/merged parity passed, or that serving is secure or production
ready. Evaluation, merge parity, CUDA execution, and deployment acceptance remain separate gates.
