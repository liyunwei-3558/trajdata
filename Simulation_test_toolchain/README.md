# Simulation Test Toolchain

This folder contains the first integrated SinD scenario testing toolchain. It is intentionally isolated from `trajdata` and `risk_mining` so test execution, policy code, and generated project outputs can evolve without changing the base dataset package.

## Run

```bash
conda activate trajdata
python -m Simulation_test_toolchain.run_test \
  --config Simulation_test_toolchain/test_projects/example_sind_single/config.yaml
```

The example runs a single SinD scene with a selected ego vehicle controlled by RiskIDM while non-ego vehicles replay ground truth. Outputs are written under `Simulation_test_toolchain/test_projects/<project_name>/`:

- `config.yaml`: copied run configuration
- `trajectory_log.json`: per-timestep agent states and ego commands
- `trajectory_log.csv`: flat trajectory log
- `interactive.html`: Bokeh timeline with map, agents, ego highlight, and ego command panel

## Policies

Supported policy names:

- `ground_truth`: replay the next state from trajdata future data
- `risk_idm`: built-in risk-aware IDM controller
- `asaprl`: ASAPRL wrapper requiring a working `torch` environment and `checkpoints.asaprl_ckpt_path`
- `diffuser`: reserved TRACE/Diffuser integration point requiring `torch`, `tbsim`, and `checkpoints.diffuser_ckpt_path`

Large checkpoint files should be placed outside git or under `Simulation_test_toolchain/checkpoints/` if ignored locally. The runner always reads checkpoint paths from YAML.

The active Python environment must be internally consistent. In particular, `trajdata` imports `pandas`, `pyarrow`, and `torch`; a NumPy 2.x environment with extensions compiled against NumPy 1.x or a mismatched CUDA PyTorch install will fail before the simulation starts.

If only one SinD location appears available, rebuild the global scene list:

```bash
conda activate trajdata
python -m Simulation_test_toolchain.rebuild_sind_scene_list \
  --data-dir datasets/SinD_dataset
```

This fixes the shared `~/.unified_data_cache/sind/scenes_list.dill` when prior single-location cache builds have overwritten it.

## Config Notes

The first version supports SinD only. Use `scenario.scene_index`, `scenario.init_timestep`, and `scenario.ego_agent_name` for deterministic inspection.

Semantic scenario labels can also drive these fields automatically:

```yaml
scenario:
  semantic_label_id: SIND_TJ_MPRTTC_8_3_4_R4_00001
  semantic_label_path: datasets/SinD_dataset/Semantic_labels/scenarios.json
  semantic_min_num_steps: 150
```

When `semantic_label_id` is set, the runner resolves `dataset.location`, `scenario.scene_index`, `scenario.init_timestep`, and `scenario.ego_agent_name` from the label. `scenario.num_steps` is expanded to cover the label window and at least `semantic_min_num_steps`.

## Semantic Label Workflow

Generate or refresh the full SinD semantic label database:

```bash
conda run -n trajdata python -m semantic_labels.import_sind_semantic_labels \
  --extract-root /media/lyw/KESU/sind-extract/SinD_Valued_Scenario_extract \
  --data-dir datasets/SinD_dataset \
  --mode full \
  --output datasets/SinD_dataset/Semantic_labels/scenarios.json
```

Validate labels against trajdata scene metadata:

```bash
conda run -n trajdata python -m semantic_labels.validate_sind_semantic_labels \
  --labels datasets/SinD_dataset/Semantic_labels/scenarios.json \
  --data-dir datasets/SinD_dataset \
  --check-trajdata
```
