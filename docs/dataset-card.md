# TicketTune synthetic support-triage datasets

TicketTune has two deliberately separate data surfaces:

- a checked-in 56-record teaching corpus for the offline quickstart and smoke tests; and
- a deterministic 1,120-record candidate for the complete local fine-tuning workflow.

Both are synthetic and CC0-dedicated. The larger candidate is generated locally and ignored by
Git; it does not become release-qualified until two distinct people independently approve every
record through the packet-backed v1.2 workflow in [qualification.md](qualification.md).

## Teaching-corpus summary

TicketTune ships a small, self-contained synthetic corpus for supervised fine-tuning of a
conversational model that triages customer-support tickets. The source file contains 56 JSONL
records. No external dataset was downloaded, transformed, or mixed into this version, and no real
customer conversation was used.

The task is deliberately structured. Given one system instruction and one user ticket, the model
must return exactly one bare JSON object with these fields. Markdown fences, surrounding prose,
additional objects, and unknown keys are contract failures:

| Field | Contract |
| --- | --- |
| `category` | `account_access`, `billing`, `bug`, `cancellation`, `feature_request`, `shipping`, or `security` |
| `priority` | `low`, `medium`, `high`, or `urgent` |
| `sentiment` | `positive`, `neutral`, `frustrated`, `angry`, or `worried` |
| `response` | A concise, safe first response that does not claim an unverified resolution |
| `next_action` | A lowercase machine-readable workflow action |

## Source schema

Every line in `data/raw/support_tickets.jsonl` is one independent record with:

- a unique `TT-####` identifier;
- exactly three ordered messages: `system`, `user`, then `assistant`;
- an `expected` object matching the assistant JSON byte-for-field after parsing;
- explicit synthetic provenance and the `CC0-1.0` dedication;
- `contains_real_customer_data: false`; and
- a complete inventory of uppercase bracketed placeholders such as `[ACCOUNT_ID]`, `[EMAIL]`,
  `[ORDER_ID]`, and `[API_KEY]`.

The loader rejects unknown fields, invalid roles, malformed or duplicate-key assistant JSON,
undeclared or unused placeholders, duplicate identifiers, duplicate normalized user content, and
common unredacted email, phone, payment-card, or US social-security-number patterns. Placeholder
values are never resolved by the project.

## Composition and balance

The corpus is exactly balanced across the requested task labels:

- seven categories with eight examples each;
- four priorities with fourteen examples each; and
- two examples for every category-priority pair.

Sentiments provide a secondary spread: 7 positive, 14 neutral, 14 frustrated, 7 angry, and 14
worried examples. Scenarios cover routine questions, recoverable failures, time-sensitive
escalations, security containment, billing disputes, cancellation state, delivery exceptions, and
feature gaps. Responses distinguish investigation or escalation from a completed fix.

## Preparation and reproducibility

`tickettune.data.prepare_dataset` performs the following offline workflow:

1. validate each JSONL line with its source line number;
2. reject ID and normalized-content duplicates before splitting;
3. resolve source/output paths and reject containment before any generated write;
4. split with a seeded, category-aware global allocator that balances priorities and sentiment
   coverage across both holdouts; small strata use an exact search, while larger strata use a
   fixed-size label-aware candidate search and bounded global frontier rather than materializing
   `C(n, k)` subsets;
5. fail closed if a feasible validation or test holdout omits a canonical category, priority, or
   sentiment label, or has imbalanced priority counts;
6. verify that IDs and normalized content do not cross splits;
7. project each record to TRL conversational `prompt` and `completion` columns so only the
   assistant completion is a training target;
8. write `train.jsonl`, `validation.jsonl`, and `test.jsonl` atomically; and
9. write `manifest.json` with the seed, fractions, IDs, label counts, exact source SHA-256, and
   exact SHA-256 for every generated split.

The checked-in profiles use a 75% / 12.5% / 12.5% split. Because each category contains eight
records, this yields six training, one validation, and one test example per category. The same
source bytes, seed, and code produce the same memberships and artifact hashes in any output
directory. Each seven-record holdout covers all five sentiments and all four priorities; priority
counts differ by at most one. Generated splits are intentionally not checked in; they are derived
artifacts.

Prepared-data verification does not trust a split hash merely because the manifest repeats it. It
revalidates the source, hashes the source before and after loading, regenerates deterministic
membership and canonical TRL projection from the configured seed/fractions, and compares manifest
counts, ordered IDs, label distributions, hashes, and requested on-disk split bytes to that
source-derived result. A source or manifest change during verification also fails closed.

## Deterministic 1,120-record candidate

`tickettune data generate-candidate --seed 42` creates exactly 1,120 unique records: seven
categories × four priorities × five sentiments × eight variants. Every record uses the same strict
schema, synthetic provenance, placeholder controls, and CC0 dedication as the teaching corpus. The
candidate is balanced across all three classification labels and is prepared into 896 training,
112 validation, and 112 frozen test records.

Current immutable evidence:

| Evidence | Value |
| --- | --- |
| Generated source SHA-256 | `611ae32a0ee3304cca87d3ab610496cc08133a5cf457239fc79574247a46f5b6` |
| Prepared manifest SHA-256 | `479bdf2dbfd93502c7416464e0ddd50fcb26f41b65d6f1eb42b208ca1fe07daa` |
| Train / validation / test | 896 / 112 / 112 |
| Automated audit A | 1,120/1,120 checks passed |
| Automated audit B | 10,080/10,080 checks passed |
| Human review | Pending; automated checks do not count as reviewers |

The generated JSONL and private reviewer packets are not distributed from Git. Reproduce the
candidate from the committed generator, then use `tickettune qualify scaffold-review` to create
two independent private review packets. Exact source, split, holdout, packet, and aggregate hashes
are rechecked before a quality-profile training directory can be allocated.

The completed local Qwen2.5 0.5B experiment passed all configured thresholds on the single frozen
112-record test and preserved all 112 outputs through a safe merge. This is bounded evidence for
the deterministic synthetic task, not proof of broad language generalization, production fitness,
human qualification, CUDA execution, or live serving. See
[`results/qualified-candidate/`](../results/qualified-candidate/README.md) for sanitized evidence.

## Intended uses

- Demonstrating LoRA or QLoRA fine-tuning on a bounded structured-generation task.
- Exercising deterministic data validation, leakage prevention, manifests, evaluation, and local
  deployment workflows without an external data license dependency.
- Comparing a base model and adapter on schema validity and exact category, priority, and sentiment
  accuracy.
- Local development, education, and portfolio review.

## Out-of-scope uses

- Production support routing without domain-specific review, larger representative data, and human
  escalation controls.
- Automated account, payment, cancellation, shipment, or security actions.
- Estimating real-world customer demographics, sentiment prevalence, or operational incident rates.
- Treating an `urgent` synthetic label as authorization for a consequential action.

## Limitations and risk controls

The examples were authored for coverage and balance, not sampled from real traffic. Their language
is cleaner and their labels are less ambiguous than live support conversations. The corpus is too
small to establish production quality, demographic fairness, multilingual behavior, or robust
resistance to prompt injection. A model can also memorize phrasing or placeholders rather than
learn the intended decision boundary.

Held-out metrics therefore demonstrate only performance on this synthetic task. Before any real
use, add independently reviewed domain data with documented consent and licensing, keep a final
test set isolated, run privacy and leakage review, evaluate adversarial and ambiguous inputs, and
require human approval for high-impact actions. Dry-run, fixture-backed, local training, local
serving, hosted serving, and production validation remain separate proof layers.

TicketTune encodes a packet-backed v1.2 gate for the separate quality profile: at least 1,000 valid
synthetic records; two distinct human reviewers; label, response, PII, and license decisions for
every record; an isolated holdout of at least 100 records; exact source/prepared/holdout hashes; and
approved packet and aggregate statuses. The legacy count-only v1.1 format is always unqualified.
Passing v1.2 is still not evidence that the corpus represents production traffic or that a trained
model meets its separate quality gates.

## Teaching-corpus version and maintenance

- Dataset version: `1.0`
- Source: `data/raw/support_tickets.jsonl`
- Source SHA-256: `c8dc4e5bc19e1230bf370a55897c1ab7d8c6dd36a2e14cb14f944a203fcdcb5f`
- Record count: 56
- License/dedication: CC0 1.0 for the synthetic records; repository code remains under the
  repository license
- Validation environment: offline local unit tests
- Mock/fixture usage: the corpus is synthetic by design; `tests/fixtures/tickets.jsonl` is a
  sixteen-record test-only subset
- Production validation status: not performed and not claimed

Any future corpus revision should update this card, retain explicit provenance, rerun the full
duplicate/leakage checks, and record new source and split hashes in a freshly generated manifest.

## Implementation and validation record

The project includes smoke, Apple Silicon, local Qwen2.5 0.5B, candidate Qwen2.5 0.5B, Qwen2.5 7B
QLoRA, review-gated Qwen2.5 7B QLoRA, and Llama 3.1 8B QLoRA profiles. Teaching-corpus preparation
still produces the deterministic 42/7/7 split used by the offline quickstart. Candidate preparation
produces the separate 896/112/112 split described above. Generated `data/processed/` bytes remain
ignored; manifests and sanitized `results/` summaries retain their immutable identities.

The closeout suite covers strict schemas, placeholder/PII controls, normalized duplicate rejection,
balanced deterministic splits, source-to-manifest chain verification, tamper rejection, training
and evaluation provenance, and deployment-plan contracts. The final aggregate test count and static
check output belong in the closeout results document so this dataset card does not become a stale
test dashboard.

The earlier 56-record CPU experiment remains historical regression evidence. The later candidate
run used a clean source revision, 336 optimizer steps, a single frozen-test evaluation, and full
112-record adapter-to-merged parity. No CUDA QLoRA, live vLLM runtime, hosted deployment, or
production-data validation is claimed.
