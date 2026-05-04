#!/usr/bin/env python
"""Build a cross-metric preliminary conclusions report for SinD risk mining."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCATION_ORDER: Tuple[str, ...] = ("cc", "tj", "cqIR", "cqNR", "cqR", "xasl")
LOCATION_NAME: Mapping[str, str] = {
    "cc": "ChangChun",
    "tj": "TianJin",
    "cqIR": "ChongQing-IR",
    "cqNR": "ChongQing-NR",
    "cqR": "ChongQing-R",
    "xasl": "XiAn-Shanglin",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def _pct(value: Any, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{100.0 * float(value):.{digits}f}%"


def _top_row(df: pd.DataFrame, column: str) -> pd.Series:
    return df.loc[pd.to_numeric(df[column], errors="coerce").idxmax()]


def _html_table(df: pd.DataFrame, columns: Sequence[Tuple[str, str, str]], max_rows: int = 20) -> str:
    rows = df.head(max_rows).to_dict("records") if not df.empty else []
    parts = ["<table><thead><tr>"]
    for _, label, _ in columns:
        parts.append(f"<th>{html.escape(label)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for key, _, kind in columns:
            value = row.get(key, "")
            if kind == "pct":
                cell = _pct(value)
            elif kind == "float":
                cell = _fmt(value)
            elif kind == "int":
                cell = _fmt(int(value)) if pd.notna(value) else ""
            else:
                cell = html.escape(str(value))
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _link(path: str, label: str) -> str:
    return f"<a href='../{html.escape(path)}'>{html.escape(label)}</a>"


def load_inputs(root: Path) -> Dict[str, pd.DataFrame]:
    return {
        "geometry": _read_csv(root / "output_geometric_topology_exposure" / "geometric_exposure_metrics.csv"),
        "density": _read_csv(root / "output_intersection_spatiotemporal_density" / "density_summary.csv"),
        "degree": _read_csv(root / "output_interaction_topology_complexity" / "interaction_degree_summary_by_location.csv"),
        "duration": _read_csv(root / "output_interaction_topology_complexity" / "interaction_duration_summary.csv"),
        "conflict": _read_csv(root / "output_conflict_patterns_spatial_clustering" / "conflict_type_summary_by_location.csv"),
        "right_overlap": _read_csv(root / "output_right_of_way_overlap" / "right_of_way_overlap_summary_by_location.csv"),
        "violations": _read_csv(root / "output_structured_violations_noncompliance" / "summary_by_location.csv"),
        "two_wheeler_class": _read_csv(root / "output_two_wheeler_heterogeneity" / "two_wheeler_summary_by_class.csv"),
        "two_wheeler_location": _read_csv(root / "output_two_wheeler_heterogeneity" / "two_wheeler_summary_by_location.csv"),
        "gap": _read_csv(root / "output_critical_gap_acceptance" / "critical_gap_summary_by_city.csv"),
        "wasserstein_pairs": _read_csv(root / "output_kinematic_wasserstein_domain_gap" / "wasserstein_city_pairs_long.csv"),
        "domain_samples": _read_csv(root / "output_kinematic_wasserstein_domain_gap" / "domain_sample_summary.csv"),
    }


def build_location_dashboard(data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    geo = data["geometry"].copy()
    keep = [
        "location",
        "conflict_area_m2",
        "mean_roi_residence_s",
        "p90_roi_residence_s",
        "n_ge_3_ratio",
        "duration_p90_s",
        "unique_spatial_overlap_cells_1m",
    ]
    dash = geo[keep].copy()
    conflict = data["conflict"][
        [
            "location",
            "total_pair_episodes",
            "crossing_ratio",
            "weaving_parallel_competition_ratio",
            "turning_interaction_ratio",
        ]
    ]
    density = data["density"][["location", "seconds_in_roi_total", "max_cell_seconds", "occupied_cells"]]
    violations = data["violations"][
        [
            "location",
            "red_light_entry_violation_rate_observable",
            "vru_encroachment_rate",
            "right_of_way_candidates",
        ]
    ]
    dash = dash.merge(conflict, on="location", how="left").merge(density, on="location", how="left").merge(violations, on="location", how="left")
    dash["city"] = dash["location"].map(LOCATION_NAME)
    dash["order"] = dash["location"].map({loc: idx for idx, loc in enumerate(LOCATION_ORDER)})
    return dash.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def build_findings(data: Mapping[str, pd.DataFrame], dash: pd.DataFrame) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []

    xasl = dash[dash["location"] == "xasl"].iloc[0]
    cq_r = dash[dash["location"] == "cqR"].iloc[0]
    cq_ir = dash[dash["location"] == "cqIR"].iloc[0]
    max_occ = _top_row(data["density"], "seconds_in_roi_total")
    max_n3 = _top_row(data["degree"], "n_ge_3_ratio")
    max_duration = _top_row(data["duration"][data["duration"].get("summary_level", "location") == "location"], "duration_p90_s")

    findings.append(
        {
            "title": "几何规模会放大暴露代价，但不是唯一解释变量",
            "evidence": (
                f"xasl 的 Ac={_fmt(xasl.conflict_area_m2)} m2、平均核心驻留={_fmt(xasl.mean_roi_residence_s)}s、"
                f"P90 驻留={_fmt(xasl.p90_roi_residence_s)}s，均处在高位；但 cqIR 的 Ac 只有 "
                f"{_fmt(cq_ir.conflict_area_m2)} m2，博弈 P90 却达到 {_fmt(cq_ir.duration_p90_s)}s。"
            ),
            "interpretation": "论文中应把几何面积作为暴露代价的放大器，同时承认信号相位、渠化方式和流量结构共同决定最终交互强度。",
        }
    )

    findings.append(
        {
            "title": "多体博弈是 SinD 路口区别于公路场景的核心复杂性",
            "evidence": (
                f"{max_n3['location']} 的 N>=3 交互比例最高，为 {_pct(max_n3['n_ge_3_ratio'])}，"
                f"最大交互阶数达到 {int(max_n3['max_degree'])}；cqIR 的长博弈最突出，P90={_fmt(max_duration['duration_p90_s'])}s。"
            ),
            "interpretation": "这支持“简单 1v1 跟驰/换道假设不足以覆盖中国路口预测问题”的论点。",
        }
    )

    findings.append(
        {
            "title": "冲突模式具有显著路口差异，不能用单一模板概括",
            "evidence": (
                f"tj/cc 的 crossing ratio 最高，分别为 {_pct(dash.loc[dash.location == 'tj', 'crossing_ratio'].iloc[0])} 和 "
                f"{_pct(dash.loc[dash.location == 'cc', 'crossing_ratio'].iloc[0])}；cqR 的 weaving/parallel competition ratio 达 "
                f"{_pct(cq_r.weaving_parallel_competition_ratio)}。"
            ),
            "interpretation": "同一套预测/规划模型跨路口迁移时，不仅面对运动学 domain gap，也面对冲突拓扑 domain gap。",
        }
    )

    row_overlap = data["right_overlap"]
    max_overlap = _top_row(row_overlap, "unique_spatial_overlap_cells_1m")
    findings.append(
        {
            "title": "合法通行本身也会生成路径重叠，路权冲突不能只按违规理解",
            "evidence": (
                f"cqR 在合法绿灯口径下有 {int(max_overlap['overlap_events'])} 个 pair-level overlap，"
                f"落在 {int(max_overlap['unique_spatial_overlap_cells_1m'])} 个 1m 网格中。"
            ),
            "interpretation": "这为“合法行驶也需要强 interaction prediction”提供了直接证据；cc/tj/xasl 的 VRU 绿灯侧当前受绑定可观测性限制。",
        }
    )

    tw = data["two_wheeler_class"].set_index("separated_class")
    ebike_speed = float(tw.loc["e_bike", "mean_passage_speed_mps"])
    bicycle_speed = float(tw.loc["bicycle", "mean_passage_speed_mps"])
    ped_speed = float(tw.loc["pedestrian", "mean_passage_speed_mps"])
    findings.append(
        {
            "title": "Powered two-wheeler 是速度接近机动车、规则属性接近 VRU 的异质主体",
            "evidence": (
                f"e-bike proxy 平均路口通行速度为 {_fmt(ebike_speed)} m/s，约为 bicycle 的 {_fmt(ebike_speed / bicycle_speed, 2)} 倍、"
                f"pedestrian 的 {_fmt(ebike_speed / ped_speed, 2)} 倍。"
            ),
            "interpretation": "这类主体强化了中国路口的异质性：运动学上更快，但空间行为和路权遵守方式更接近非结构化交通参与者。",
        }
    )

    wasserstein = data["wasserstein_pairs"].copy()
    non_self = wasserstein[wasserstein["domain_a"] != wasserstein["domain_b"]]
    max_pair = _top_row(non_self, "wasserstein_2d_normalized")
    min_pair = non_self.loc[pd.to_numeric(non_self["wasserstein_2d_normalized"], errors="coerce").idxmin()]
    findings.append(
        {
            "title": "城市间运动学分布存在可量化 domain gap",
            "evidence": (
                f"2D Wasserstein 最大 pair 为 {max_pair['domain_a']} vs {max_pair['domain_b']}，W={_fmt(max_pair['wasserstein_2d_normalized'])}；"
                f"最小非零 pair 为 {min_pair['domain_a']} vs {min_pair['domain_b']}，W={_fmt(min_pair['wasserstein_2d_normalized'])}。"
            ),
            "interpretation": "这支持将跨城市模型迁移视为真实 domain adaptation 问题，而不是简单数据量扩大问题。",
        }
    )

    violations = data["violations"]
    red_top = _top_row(violations, "red_light_entry_violation_rate_observable")
    vru_top = _top_row(violations, "vru_encroachment_rate")
    findings.append(
        {
            "title": "规则背景噪声不可忽略，但部分指标仍应作为候选/代理解释",
            "evidence": (
                f"可观测红灯入口率最高的是 {red_top['location']}，为 {_pct(red_top['red_light_entry_violation_rate_observable'])}；"
                f"VRU encroachment rate 最高的是 {vru_top['location']}，为 {_pct(vru_top['vru_encroachment_rate'])}。"
            ),
            "interpretation": "这些结果可用于说明 rule-based 假设的脆弱性，但红灯、VRU、路权等指标依赖信号绑定和地图语义，应在论文中保留可观测性说明。",
        }
    )

    findings.append(
        {
            "title": "驻留热力与冲突拓扑提供互补证据",
            "evidence": (
                f"总驻留秒数最高的是 {max_occ['location']}，为 {_fmt(max_occ['seconds_in_roi_total'], 1)}s；"
                f"最大单元格驻留也在 {max_occ['location']}，为 {_fmt(max_occ['max_cell_seconds'], 1)}s。"
            ),
            "interpretation": "热力图定位“车辆在哪里停留/博弈”，交互图解释“有多少主体在一起博弈”，两者应成组展示。",
        }
    )

    return findings


def plot_key_bars(dash: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), dpi=180)
    specs = [
        ("mean_roi_residence_s", "Mean Core Residence (s)", "#2f6f8f"),
        ("n_ge_3_ratio", "N>=3 Interaction Ratio", "#8a4f9e"),
        ("weaving_parallel_competition_ratio", "Weaving / Parallel Competition", "#d18f00"),
        ("red_light_entry_violation_rate_observable", "Observable Red Entry Rate", "#c0392b"),
    ]
    for ax, (col, title, color) in zip(axes.ravel(), specs):
        ax.bar(dash["location"], dash[col], color=color, alpha=0.82)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.24)
        if "ratio" in col or "rate" in col:
            ax.set_ylim(0, max(0.05, float(dash[col].max()) * 1.15))
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_conclusion_matrix(dash: pd.DataFrame, output_path: Path) -> None:
    cols = [
        "conflict_area_m2",
        "mean_roi_residence_s",
        "n_ge_3_ratio",
        "duration_p90_s",
        "crossing_ratio",
        "weaving_parallel_competition_ratio",
        "vru_encroachment_rate",
    ]
    labels = ["Ac", "Residence", "N>=3", "P90 game", "Crossing", "Weaving", "VRU encroach"]
    mat = dash[cols].astype(float).copy()
    normalized = (mat - mat.min()) / (mat.max() - mat.min()).replace(0, np.nan)
    normalized = normalized.fillna(0.0).to_numpy()
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=180)
    im = ax.imshow(normalized, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(dash)))
    ax.set_yticklabels(dash["location"])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#102028")
    ax.set_title("Normalized Cross-Metric Profile by Intersection")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="0-1 normalized")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_report_html(output_path: Path, findings: Sequence[Mapping[str, str]], dash: pd.DataFrame, data: Mapping[str, pd.DataFrame]) -> None:
    finding_cards = "\n".join(
        f"""
        <article class="finding">
          <h3>{idx}. {html.escape(item['title'])}</h3>
          <p><b>Evidence.</b> {html.escape(item['evidence'])}</p>
          <p><b>Interpretation.</b> {html.escape(item['interpretation'])}</p>
        </article>
        """
        for idx, item in enumerate(findings, start=1)
    )
    dash_table = _html_table(
        dash,
        [
            ("location", "Loc", "text"),
            ("conflict_area_m2", "Ac", "float"),
            ("mean_roi_residence_s", "Mean res.", "float"),
            ("p90_roi_residence_s", "P90 res.", "float"),
            ("n_ge_3_ratio", "N>=3", "pct"),
            ("duration_p90_s", "P90 game", "float"),
            ("crossing_ratio", "Crossing", "pct"),
            ("weaving_parallel_competition_ratio", "Weaving", "pct"),
            ("red_light_entry_violation_rate_observable", "Red entry", "pct"),
            ("vru_encroachment_rate", "VRU enc.", "pct"),
        ],
    )
    tw_table = _html_table(
        data["two_wheeler_class"],
        [
            ("separated_class", "Class", "text"),
            ("tracks", "Tracks", "int"),
            ("mean_passage_speed_mps", "Mean speed", "float"),
            ("mean_startup_accel_mps2", "Startup accel", "float"),
        ],
    )
    links = [
        ("output_geometric_topology_exposure/index.html", "Geometry & Exposure"),
        ("output_intersection_spatiotemporal_density/index.html", "Spatiotemporal Density"),
        ("output_interaction_topology_complexity/index.html", "Interaction Topology"),
        ("output_conflict_patterns_spatial_clustering/index.html", "Conflict Patterns"),
        ("output_kinematic_wasserstein_domain_gap/index.html", "Wasserstein Domain Gap"),
        ("output_two_wheeler_heterogeneity/index.html", "Two-Wheeler Heterogeneity"),
        ("output_right_of_way_overlap/index.html", "Right-of-Way Overlap"),
        ("output_structured_violations_noncompliance/index.html", "Violations & Non-compliance"),
    ]
    link_html = "".join(f"<li>{_link(path, label)}</li>" for path, label in links)
    output_path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SinD 指标初步结论分析</title>
  <style>
    body {{ margin:0; font-family:Arial, sans-serif; background:#f4f1eb; color:#172126; }}
    header {{ padding:34px 5vw; background:#203542; color:white; }}
    main {{ padding:24px 5vw 54px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; }}
    .panel,.finding {{ background:white; border:1px solid #d8d0c4; padding:18px; margin:18px 0; }}
    .finding h3 {{ margin-top:0; color:#17394a; }}
    .finding p {{ line-height:1.58; }}
    img {{ max-width:100%; border:1px solid #d8d0c4; background:white; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ border-bottom:1px solid #e5ded3; padding:8px 9px; text-align:left; white-space:nowrap; }}
    th {{ background:#ede4d8; }}
    .note {{ color:#5d6870; line-height:1.55; }}
    a {{ color:#105d7a; }}
  </style>
</head>
<body>
  <header>
    <h1>SinD 已提取指标初步结论分析</h1>
    <p>综合几何暴露、运动学分布、多体交互、冲突模式、规则背景噪声和两轮车异质性。</p>
  </header>
  <main>
    <section class="panel">
      <p class="note">本报告是论文讨论草稿级结论。部分指标是 candidate/proxy，且交通灯与 crosswalk 绑定仍存在可观测性差异；正式表述应保留这些边界。</p>
      <ul>{link_html}</ul>
    </section>
    <section class="grid">
      <div class="panel"><a href="conclusion_key_bars.png"><img src="conclusion_key_bars.png" alt="key bars"></a></div>
      <div class="panel"><a href="conclusion_metric_matrix.png"><img src="conclusion_metric_matrix.png" alt="metric matrix"></a></div>
    </section>
    <section class="panel"><h2>主要结论</h2>{finding_cards}</section>
    <section class="panel"><h2>六路口核心指标对照</h2>{dash_table}</section>
    <section class="panel"><h2>两轮车动力学差异</h2>{tw_table}</section>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-mining-dir", type=Path, default=Path("risk_mining"))
    parser.add_argument("--output-dir", type=Path, default=Path("risk_mining/output_metric_conclusions"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_inputs(args.risk_mining_dir)
    dash = build_location_dashboard(data)
    findings = build_findings(data, dash)
    dash.to_csv(args.output_dir / "conclusion_location_dashboard.csv", index=False)
    pd.DataFrame(findings).to_csv(args.output_dir / "preliminary_findings.csv", index=False)
    plot_key_bars(dash, args.output_dir / "conclusion_key_bars.png")
    plot_conclusion_matrix(dash, args.output_dir / "conclusion_metric_matrix.png")
    build_report_html(args.output_dir / "index.html", findings, dash, data)
    print(f"Done. Open {args.output_dir / 'index.html'}")
    for idx, item in enumerate(findings, start=1):
        print(f"{idx}. {item['title']}")


if __name__ == "__main__":
    main()
