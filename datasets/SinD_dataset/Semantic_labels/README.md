# SinD Semantic Labels

This directory stores normalized semantic labels for short SinD scenario slices.
Each label points back to the original trajdata scene, frame window, ego agent,
semantic tags, and source extraction file.

Current full import statistics:

- total: 53901
- high_risk_mprttc: 31005
- visual_shielding: 1286
- narrow_feasible_area: 21610

## Files

- `schema.json`: JSON schema for the label file.
- `scenarios.json`: generated semantic label database.

## Generate

Full import:

```bash
conda run -n trajdata python -m semantic_labels.import_sind_semantic_labels \
  --extract-root /media/lyw/KESU/sind-extract/SinD_Valued_Scenario_extract \
  --data-dir datasets/SinD_dataset \
  --mode full \
  --output datasets/SinD_dataset/Semantic_labels/scenarios.json
```

Quick sample import:

```bash
conda run -n trajdata python -m semantic_labels.import_sind_semantic_labels \
  --extract-root /media/lyw/KESU/sind-extract/SinD_Valued_Scenario_extract \
  --data-dir datasets/SinD_dataset \
  --mode sample \
  --max-narrow-scenes-per-location 3 \
  --max-narrow-parts-per-scene 1 \
  --output datasets/SinD_dataset/Semantic_labels/scenarios.json
```

## Validate

```bash
conda run -n trajdata python -m semantic_labels.validate_sind_semantic_labels \
  --labels datasets/SinD_dataset/Semantic_labels/scenarios.json \
  --data-dir datasets/SinD_dataset \
  --check-trajdata
```

## Test Toolchain Use

Set `scenario.semantic_label_id` in a `Simulation_test_toolchain` config. The
runner resolves `location`, `scene_index`, `init_timestep`, and `ego_agent_name`
from the label before simulation.
