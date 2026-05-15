from __future__ import annotations

import numpy as np

from Simulation_test_toolchain.run_qcnet_prediction_eval import (
    compute_prediction_metrics,
)


def test_compute_prediction_metrics_top1_and_min_of_k():
    gt = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    pred = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    probs = np.asarray([0.7, 0.2, 0.1], dtype=np.float32)
    valid = np.asarray([True, True, True])

    metrics, details = compute_prediction_metrics(
        pred_modes_world=pred,
        probs=probs,
        gt_world=gt,
        valid_mask=valid,
        max_guesses=3,
        miss_threshold=2.0,
    )

    assert details["top1_mode_index"] == 0
    assert details["best_mode_index"] == 1
    assert metrics["top1_ADE"] == 2.0
    assert metrics["top1_FDE"] == 4.0
    assert metrics["top1_MR"] == 1.0
    assert metrics["minADE"] == 0.0
    assert metrics["minFDE"] == 0.0
    assert metrics["MR"] == 0.0
