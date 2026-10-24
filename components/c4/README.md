

## Revised replay: supplementary required files

In addition to the revised replay bundle, the default case requires these files beneath `c4_artifacts/benchmark_runs/adult/adult-hybrid-fresh-validation-v1/adult-ns51001-r004-interaction-low-broadened_slice-label_corrupted_repair/`:

- `forecast_freeze-v1/controller_decisions.json`
- `pre_release_freeze-v1/upstream_simulation.json`
- `pre_release_freeze-v1/upstream_simulation.sha256`
- `scoped_candidate_validation-v1/metadata.json`
- `scoped_candidate_validation-v1/metadata.sha256`
- `scoped_candidate_validation-v1/validation.json`
- `scoped_candidate_validation-v1/validation.sha256`

Preserve original bytes and SHA256 bindings. Use extended Windows paths when packaging deeply nested artifacts. ZIP integrity alone does not establish dependency completeness.
