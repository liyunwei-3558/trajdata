#!/usr/bin/env python
"""Export the main SinD risk-mining reports into a single self-contained HTML file."""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote

REPORT_ORDER: Tuple[Tuple[str, str, str], ...] = (
    ("output_metric_conclusions", "Metric Conclusions", "Cross-metric executive summary."),
    ("output_geometric_topology_exposure", "Geometric Topology & Exposure", "Area scale vs exposure cost and topology trends."),
    ("output_intersection_spatiotemporal_density", "Intersection Spatiotemporal Density", "Core ROI heatmaps and residence statistics."),
    ("output_interaction_topology_complexity", "Interaction Topology & Game Complexity", "Multi-agent interaction degree and duration."),
    ("output_conflict_patterns_spatial_clustering", "Conflict Patterns & Spatial Clustering", "Conflict type ratios and hotspot maps."),
    ("output_right_of_way_overlap", "Right-of-Way Overlap", "Legal green-phase path overlap evidence."),
    ("output_structured_violations_noncompliance", "Structured Violations & Non-compliance", "Vehicle and VRU rule-background indicators."),
    ("output_two_wheeler_heterogeneity", "Two-Wheeler Heterogeneity", "E-bike vs bicycle speed and acceleration split."),
    ("output_lateral_deviation_variance", "Lateral Deviation Variance", "Bundle overlays and lane-relative offsets."),
    ("output_critical_gap_acceptance", "Critical Gap Acceptance", "Gap acceptance and aggressiveness baselines."),
    ("output_kinematic_envelopes_six_all", "Kinematic Envelopes", "City-wise v-a envelopes and cross-city overlap."),
    ("output_kinematic_wasserstein_domain_gap", "Wasserstein Domain Gap", "Pairwise motion-distribution distance matrices."),
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
EMBED_LIMIT_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class EmbeddedReport:
    slug: str
    title: str
    note: str
    standalone_html: str
    source_path: Path


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime:
        return mime
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "text/csv"
    if suffix == ".json":
        return "application/json"
    if suffix == ".html":
        return "text/html"
    if suffix == ".svg":
        return "image/svg+xml"
    return "application/octet-stream"


def _to_data_uri(path: Path) -> str:
    mime = _guess_mime(path)
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _resolve_local_path(base_dir: Path, url: str) -> Optional[Path]:
    cleaned = unquote(url).strip()
    if not cleaned or cleaned.startswith("#"):
        return None
    if cleaned.startswith(("http://", "https://", "data:", "mailto:", "javascript:", "//")):
        return None
    path = (base_dir / cleaned).resolve()
    try:
        path.relative_to(base_dir.resolve())
    except ValueError:
        return None
    return path if path.exists() and path.is_file() else None


def _rewrite_local_images(text: str, base_dir: Path) -> str:
    def repl(match: re.Match[str]) -> str:
        prefix, url, suffix = match.group(1), match.group(2), match.group(3)
        path = _resolve_local_path(base_dir, url)
        if path is None:
            return match.group(0)
        if path.suffix.lower() not in IMAGE_EXTS and path.stat().st_size > EMBED_LIMIT_BYTES:
            return match.group(0)
        return f"{prefix}{_to_data_uri(path)}{suffix}"

    return re.sub(r'(<img\b[^>]*?\bsrc=["\'])([^"\']+)(["\'])', repl, text, flags=re.IGNORECASE)


def _rewrite_local_hrefs(text: str, base_dir: Path) -> str:
    def repl(match: re.Match[str]) -> str:
        prefix, url, suffix = match.group(1), match.group(2), match.group(3)
        path = _resolve_local_path(base_dir, url)
        if path is None:
            return match.group(0)
        if path.suffix.lower() in IMAGE_EXTS or path.stat().st_size <= EMBED_LIMIT_BYTES:
            return f"{prefix}{_to_data_uri(path)}{suffix}"
        return f"{prefix}#{suffix}"

    return re.sub(r'(<a\b[^>]*?\bhref=["\'])([^"\']+)(["\'])', repl, text, flags=re.IGNORECASE)


def _standaloneize_report(report_path: Path) -> str:
    text = report_path.read_text(encoding="utf-8")
    base_dir = report_path.parent
    text = _rewrite_local_images(text, base_dir)
    text = _rewrite_local_hrefs(text, base_dir)
    return text


def _embed_reports(repo_root: Path) -> List[EmbeddedReport]:
    reports: List[EmbeddedReport] = []
    for slug, title, note in REPORT_ORDER:
        report_path = repo_root / "risk_mining" / slug / "index.html"
        if not report_path.exists():
            continue
        reports.append(
            EmbeddedReport(
                slug=slug,
                title=title,
                note=note,
                standalone_html=_standaloneize_report(report_path),
                source_path=report_path,
            )
        )
    return reports


def _build_page(reports: Sequence[EmbeddedReport]) -> str:
    cards = []
    for idx, report in enumerate(reports, start=1):
        encoded = base64.b64encode(report.standalone_html.encode("utf-8")).decode("ascii")
        cards.append(
            f"""
            <section class="report-card" id="{html.escape(report.slug)}">
              <div class="report-head">
                <div>
                  <div class="report-kicker">Report {idx:02d}</div>
                  <h2>{html.escape(report.title)}</h2>
                  <p>{html.escape(report.note)}</p>
                </div>
                <div class="report-path">{html.escape(str(report.source_path))}</div>
              </div>
              <iframe loading="lazy" src="data:text/html;base64,{encoded}" title="{html.escape(report.title)}"></iframe>
            </section>
            """
        )
    toc = "\n".join(
        f'<a href="#{html.escape(report.slug)}">{idx:02d}. {html.escape(report.title)}</a>'
        for idx, report in enumerate(reports, start=1)
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SinD Standalone Report Bundle</title>
  <style>
    :root {{
      --bg:#f4efe5; --ink:#1f272a; --card:#fffaf2; --line:#d9ccb6;
      --accent:#8b3d2e; --accent2:#24586b; --muted:#6d6a62;
    }}
    body {{ margin:0; font-family:Arial, sans-serif; color:var(--ink); background:linear-gradient(135deg,#f4efe5,#e7eee9); }}
    header {{ padding:36px 5vw 28px; background:#24363f; color:#fff; }}
    header h1 {{ margin:0 0 10px; font-size:clamp(28px,4.8vw,52px); letter-spacing:-.8px; }}
    header p {{ margin:0; max-width:1100px; line-height:1.65; color:#e9dfd3; font-size:16px; }}
    main {{ padding:22px 5vw 48px; }}
    .summary {{
      display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px;
      margin-top:-34px; margin-bottom:24px;
    }}
    .stat, .report-card, .toc, .note {{
      background:var(--card); border:1px solid var(--line); border-radius:18px;
      box-shadow:0 12px 28px rgba(31,39,42,.08);
    }}
    .stat {{ padding:16px 18px; }}
    .stat b {{ display:block; margin-top:6px; color:var(--accent); font-size:24px; }}
    .stat small {{ color:var(--muted); }}
    .toc {{ padding:16px 18px; margin-bottom:22px; }}
    .toc h2 {{ margin:0 0 12px; font-size:20px; }}
    .toc a {{ display:inline-block; margin:6px 10px 0 0; padding:8px 12px; border-radius:999px; text-decoration:none; color:var(--accent2); border:1px solid var(--line); background:#fff; }}
    .note {{ padding:14px 18px; margin-bottom:22px; color:var(--muted); line-height:1.6; }}
    .report-card {{ padding:16px; margin:20px 0; }}
    .report-head {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-start; margin-bottom:12px; }}
    .report-kicker {{ text-transform:uppercase; letter-spacing:.12em; font-size:11px; color:var(--muted); }}
    .report-head h2 {{ margin:6px 0 6px; font-size:26px; }}
    .report-head p {{ margin:0; color:var(--muted); line-height:1.55; }}
    .report-path {{ color:var(--accent2); font-size:12px; text-align:right; word-break:break-all; max-width:42%; }}
    iframe {{ width:100%; height:88vh; border:1px solid var(--line); border-radius:14px; background:#fff; }}
    @media (max-width: 800px) {{
      .report-head {{ flex-direction:column; }}
      .report-path {{ max-width:100%; text-align:left; }}
      iframe {{ height:80vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>SinD Standalone Report Bundle</h1>
    <p>单文件离线版总览，内嵌了现有分析报告的页面内容与图片资源。把这一份 HTML 拷贝到别的电脑上，直接打开即可浏览。</p>
  </header>
  <main>
    <section class="summary">
      <div class="stat"><small>Embedded reports</small><b>{len(reports)}</b></div>
      <div class="stat"><small>Offline mode</small><b>standalone HTML</b></div>
      <div class="stat"><small>Assets</small><b>embedded</b></div>
      <div class="stat"><small>Target use</small><b>portable review</b></div>
    </section>
    <section class="toc">
      <h2>Table of Contents</h2>
      {toc}
    </section>
    <section class="note">
      说明：本文件优先嵌入图片与轻量本地链接。极大的原始 CSV/JSON 等可在报告内保留为占位链接，但不会强行塞进单文件，以免 HTML 体积失控。
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("risk_mining/output_standalone_reports/index.html"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    reports = _embed_reports(repo_root)
    if not reports:
        raise RuntimeError("No report HTML files found to embed.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page = _build_page(reports)
    output_path.write_text(page, encoding="utf-8")
    print(f"Wrote standalone bundle to: {output_path}")
    print(f"Embedded reports: {len(reports)}")


if __name__ == "__main__":
    main()
