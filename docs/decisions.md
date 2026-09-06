# Decision register

Date of initial capture: 2026-09-06. Earlier direction is reconstructed from the 2026-09-04 KRAFT discussion and the supplied project context; missing details remain open.

| ID | Status | Decision | Rationale / consequence |
| --- | --- | --- | --- |
| D001 | Agreed | Name: KRAFT | Reference to description-length priors and the Kraft inequality |
| D002 | Agreed | Rust, CPU first, with a future GPU path | Correctness and enumeration first; explicit packed state and batched kernels later |
| D003 | Agreed direction | Start with finite-state / small integer transition objects | Avoid VM overhead in the first investigation; exact tiny spaces are possible |
| D004 | Agreed direction | Search prioritization uses description length plus prediction loss | Posterior contribution is the organizing quantity; longer programs may enter later |
| D005 | Mathematical clarification | Uniform change of log base adds no scheduling degree of freedom | `u/c` is unchanged; a compute exponent is a separate policy knob |
| D006 | Proposed in bootstrap | Exact oracle before adaptive scheduling | Separates inference correctness from search-policy quality |
| D007 | Bootstrap implementation | Single dependency-free Rust crate; publication disabled | Avoid speculative workspace/framework complexity; split modules when implemented |
| D008 | Proposed in bootstrap | Binary synthetic sequences and probabilistic finite-state emissions for E0 | Small model spaces and analytic examples; exact emission scheme still needs Q1 |
| D009 | Bootstrap implementation | Research prose owns theory/results; code/config owns executable defaults | Reduce documentation drift |

## Open decisions

- O1: transition convention, emissions, start state, model bounds, and parameter priors.
- O2: actual self-delimiting code and model-length prior; how finite truncation is normalized.
- O3: labeled descriptions versus canonicalized models with aggregated prior mass.
- O4: multiply-shift formula, word width, overflow, state range, and structural capacity.
- O5: scheduling cost unit and policy; whether fixed speed-weighted priors merit a separate comparison.
- O6: exact shared-likelihood lower bounds for frontier pruning.
- O7: artifact backend and eventual experiment tracking integration.
- O8: license. No license grant was selected during bootstrap.

For changes, append a dated record stating evidence, alternatives, consequences, and superseded IDs. Do not silently rewrite an earlier accepted decision as though it had always been the plan.
