# Research notebook

Start with the [byte harness](harness.md) to run the current code. Read these in order for a complete handoff:

1. [Status](status.md): what exists, what has been tested, and the immediate next step.
2. [Prequential objective](prequential.md): the canonical KRAFT score and causal coding contract.
3. [Theory](theory.md): priors, inference bounds, DFA mathematics, and research scope.
4. [Decisions](decisions.md): accepted direction, provisional choices, and open decisions.
5. [Architecture](architecture.md): implementation boundaries for the anytime Bayesian learner.
6. [Methodology](methodology.md): how evidence will be collected and compared.
7. [Experiments](experiments/README.md): ordered hypotheses and success gates.
8. [Queue](queue.md): actionable work and dependencies.
9. [Research log](research-log.md): dated history, including negative results.

[References](references.md) tracks background reading. Use the [experiment template](templates/experiment.md) for each run campaign.

## Maintenance contract

| Record | Update when | Owns |
| --- | --- | --- |
| Status | A milestone, blocker, or capability changes | Current snapshot |
| Research log | Substantive work is completed or abandoned | Chronological history |
| Decisions | A consequential choice is made or revised | Rationale and alternatives |
| Methodology | Evaluation protocol changes | Shared experimental rules |
| Experiment record | A campaign is planned, run, or analyzed | Protocol, evidence, limitations |
| Queue | A task becomes ready, blocked, or complete | Future actions and dependencies |

Keep one authoritative home for each fact. Link from status/log to detailed results. Never turn a hypothesis into a finding without a run record. Log dates use UTC; experiment records identify execution time separately from write-up time.
