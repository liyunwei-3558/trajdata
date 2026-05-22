# RIDM / ASAPRL / QCNet Semantic High-Risk Metrics Summary

Scope: dataset-open-loop policy evaluation on three semantic high-risk scenario families: `MprTTC`, `visual_shielding`, and `narrow_feasible_area`. Non-ego agents replay ground truth. Metrics are aggregated from `run_manifest.csv` files.

## Coverage

| Scenario | Policy | Planned | Completed | Failed | Skipped | Source | Note |
|---|---:|---:|---:|---:|---:|---|---|
| MprTTC | RIDM | 1500 | 1500 | 0 | 0 | `mprttc_riskidm_1500_merged/raw_open_loop_risk_idm` | merged 1500 |
| MprTTC | ASAPRL | 1500 | 1500 | 0 | 0 | `mprttc_asaprl_1500_merged/raw_open_loop_asaprl` | merged 1500 |
| MprTTC | QCNet | 1200 | 1200 | 0 | 0 | `qcnet_semantic_mprttc_per_intersection_200/raw_open_loop_qcnet` | 200/intersection, 1200 total |
| visual_shielding | RIDM | - | - | - | - | `` | not found |
| visual_shielding | ASAPRL | - | - | - | - | `` | not found |
| visual_shielding | QCNet | 1000 | 1000 | 0 | 0 | `qcnet_semantic_visual_shielding_forced_1000/raw_open_loop_qcnet` | forced 1000; valid-window requirements relaxed |
| narrow_feasible_area | RIDM | 1000 | 1000 | 0 | 0 | `narrow_feasible_riskidm_per_intersection_100pairs_fast_corrected/raw_open_loop_risk_idm` | 100 pairs/intersection; original+boosted |
| narrow_feasible_area | ASAPRL | 1000 | 1000 | 0 | 0 | `narrow_feasible_asaprl_per_intersection_100pairs/raw_open_loop_asaprl/shard_00+shard_01` | 100 pairs/intersection; original+boosted; two shards merged in this report |
| narrow_feasible_area | QCNet | 1000 | 1000 | 0 | 0 | `qcnet_semantic_narrow_feasible_per_intersection_200/raw_open_loop_qcnet` | 100 pairs/intersection; original+boosted |

## Overall Metrics

| Scenario | Policy | ADE mean/med | FDE mean/med | MinTTC mean/med | AveTTC mean | MRD mean | ARD mean | Collision | Offroad | Violation | Red-light | Wrong-way | Lane-dir | Anomaly |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MprTTC | RIDM | 0.47/0.04 | 1.47/0.03 | 1.62/2.50 | 2.14 | 1.97 | 2.93 | 30.3% | 0.2% | 5.9% | 5.2% | 0.4% | 0.8% | 1.4% |
| MprTTC | ASAPRL | 11.41/9.54 | 24.44/23.00 | 0.97/0.00 | 1.65 | 1.45 | 5.72 | 55.1% | 15.6% | 31.0% | 26.7% | 0.5% | 6.4% | 49.0% |
| MprTTC | QCNet | 2.09/0.08 | 6.32/0.12 | 0.03/0.00 | 1.13 | 0.00 | 0.79 | 98.3% | 0.8% | 6.8% | 5.1% | 1.5% | 1.4% | 6.7% |
| visual_shielding | RIDM | -/- | -/- | -/- | - | - | - | - | - | - | - | - | - | - |
| visual_shielding | ASAPRL | -/- | -/- | -/- | - | - | - | - | - | - | - | - | - | - |
| visual_shielding | QCNet | 11.17/9.30 | 34.64/30.99 | 0.34/0.00 | 1.66 | 0.14 | 3.91 | 81.8% | 21.4% | 38.0% | 24.8% | 13.0% | 6.3% | 45.5% |
| narrow_feasible_area | RIDM | 6.26/4.92 | 9.04/3.98 | 0.63/0.00 | 1.76 | 0.40 | 1.67 | 53.4% | 10.7% | 14.2% | 12.1% | 0.3% | 2.2% | 20.2% |
| narrow_feasible_area | ASAPRL | 5.94/4.21 | 12.36/8.44 | 0.49/0.00 | 1.54 | 0.29 | 2.14 | 62.5% | 10.7% | 22.7% | 19.4% | 2.0% | 8.7% | 16.6% |
| narrow_feasible_area | QCNet | 32.45/27.80 | 71.39/64.38 | 0.25/0.00 | 1.70 | 0.19 | 9.11 | 86.8% | 64.0% | 52.9% | 34.5% | 29.5% | 13.8% | 84.7% |

## Narrow Feasible Area: Speed-Mode Split

| Policy | Speed mode | N | ADE mean | FDE mean | Collision | Offroad | Violation | Anomaly |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RIDM | boosted_speed | 500 | 6.15 | 8.27 | 58.8% | 10.8% | 16.0% | 17.8% |
| RIDM | original_speed | 500 | 6.38 | 9.81 | 48.0% | 10.6% | 12.4% | 22.6% |
| ASAPRL | boosted_speed | 500 | 6.05 | 12.50 | 65.4% | 10.8% | 23.6% | 15.8% |
| ASAPRL | original_speed | 500 | 5.83 | 12.23 | 59.6% | 10.6% | 21.8% | 17.4% |
| QCNet | boosted_speed | 500 | 48.70 | 105.57 | 92.6% | 95.6% | 56.4% | 99.4% |
| QCNet | original_speed | 500 | 16.20 | 37.22 | 81.0% | 32.4% | 49.4% | 70.0% |

## Notes

- `Anomaly` is recomputed uniformly as `ADE > 10` or `FDE > 50` for completed runs.
- `visual_shielding` currently has no RIDM/ASAPRL semantic batch output under `Simulation_test_toolchain/batch_outputs`; only QCNet forced-1000 results are available.
- The `visual_shielding` QCNet row uses the forced 1000-run batch with valid-window requirements relaxed; it should be treated as a stress-test setting.
- MprTTC sample counts differ by policy: RIDM/ASAPRL use merged 1500-run outputs, while QCNet uses the 1200-run `200/intersection` semantic batch.
- `narrow_feasible_area` uses 100 candidate pairs per intersection for cc/cqIR/cqNR/cqR/xa, expanded into `original_speed` and `boosted_speed` runs; tj is not included.
