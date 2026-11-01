# Model card: CascadeLens ranker

**Status: {{status}}.** No config is frozen. Numbers come from committed result JSONs, traced in
[provenance](docs/results/provenance.md).

## Intended use

A read-only decision aid that gives engineers a {{disclaimer}}, from metrics of a single incident
window, with a confidence and an abstention flag.

## Out of scope

- Not an autonomous operator. It must not trigger remediation.
- Not proof of causation, and not causal discovery.
- Not validated on production data. Results come from injected benchmark faults.
- Not a model of changing topology.
- Not an internet-facing service. The API has no authentication.

## Model

A LightGBM retrieval stage followed by a LightGBM LambdaRank reranker over per-service features:
signal, temporal, graph, structural and mask. The CascadeLens ranker (`ranker_augmented`) also
trains on copies with simulated telemetry loss. Service names are not features.

## Training data

RE1 from RCAEval: {{n_cases}} metrics-only fault-injection cases across Online Boutique, Sock Shop
and Train Ticket. Splits keep each injection run and its repetitions together. RE1 records no
topology, so graphs are inferred from pre-incident latency and have not been checked against real
call graphs.

The `diagnose` command and API serve a model saved for one training fold, the plain two-stage ranker
by default. They do not serve a model trained on all data.

## Metrics

Clean group_run Top-1 for `ranker_augmented`: {{ranker_clean}}. Plain ranker: {{plain_clean}}. Flat
LightGBM, the best baseline: {{flat_clean}}.

{{table:headline}}

{{table:clean_supervised}}

Robustness, simulated: at 30% failure-correlated loss `ranker_augmented` scores {{ranker_p2}}, and
flat LightGBM scores {{flat_p2}}.

Held-out:

{{table:failure_system}}

## Calibration

Temperature scaling or isotonic regression, chosen per fold on validation only. ECE on clean
group_run falls from {{ece_before}} to {{ece_after}}, and Brier from {{brier_before}} to
{{brier_after}}. At 50% coverage Top-1 is {{top1_50}}, against {{top1_full}} at full coverage.

{{table:coverage_drift}}

## Known failure modes

- Abstention coverage is unstable on unseen fault types and systems, so a 50% target may give far
  more or far less.
- Simulated P2 loss and a 120 s early window shift cut coverage to {{coverage_p2}}% and
  {{coverage_shift}}%, and raise ECE to {{ece_p2}} and {{ece_shift}}.
- Folds with no validation errors give the calibrator nothing to learn from, so confidence
  saturates. A confidence of 1.00 is not certainty.
- Retrieval can miss the true service, which then cannot be ranked first. Recall at the chosen K is
  {{retrieval_group_run}}% on group_run and {{retrieval_loso}}% on LOSO.
- The mask-only probe reaches {{mask_only}} on clean data, against a floor of {{mask_floor}}, so
  missingness may reflect collection artifacts.
- Simulated P2 gaps depend on ground truth, and mask features gain {{mask_gain}} points there, which
  may be the gaps revealing the fault rather than robustness.

## Ethical considerations

A wrong ranking can send responders the wrong way during an incident, so a person must check the
evidence. The tool makes no outbound request and changes nothing. It is for defensive use. The
dataset is not redistributed. [VERIFY: RCAEval dataset license terms]
