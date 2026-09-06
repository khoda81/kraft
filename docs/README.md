# Research notebook

Read these in order for a complete handoff:

1. [Status](status.md): what exists, what has been tested, and the immediate next step.
2. [Theory](theory.md): the project question and mathematical commitments.
3. [Decisions](decisions.md): accepted direction, provisional choices, and open decisions.
4. [Architecture](architecture.md): implementation boundaries and the CPU-to-GPU path.
5. [Methodology](methodology.md): how evidence will be collected and compared.
6. [Experiments](experiments/README.md): ordered hypotheses and success gates.
7. [Queue](queue.md): actionable work and dependencies.
8. [Research log](research-log.md): dated history, including negative results.

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
