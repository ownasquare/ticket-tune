# Synthetic dataset license

The synthetic records in `data/raw/support_tickets.jsonl`, the derived test fixture in
`tests/fixtures/tickets.jsonl`, and the deterministic candidate produced by
`tickettune data generate-candidate` are dedicated to the public domain under the
[Creative Commons CC0 1.0 Universal dedication](https://creativecommons.org/publicdomain/zero/1.0/).

The 1,120-record candidate is generated locally at `data/qualified/support_tickets.jsonl` and is
intentionally not committed. Its public evidence is the generator source, exact SHA-256 values,
aggregate audit reports, and review workflow. Human review packets remain private and are not part
of the public dataset distribution.

This dedication applies only to those synthetic dataset records. TicketTune source code remains
under the MIT License in `LICENSE`; model weights, tokenizers, and third-party tools retain their
upstream terms. CC0 provides the material without warranties or conditions.
