"""Portable reports generated solely from stored experimental evidence."""
from __future__ import annotations

import csv
import io
import json
import math
from html import escape
from typing import Any


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _show(value: Any) -> str:
    if value is None:
        return "Unavailable / unbounded"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True, allow_nan=False)
    return str(value)


def _flatten(value: Any, prefix: str = ""):
    if isinstance(value, dict) and value:
        for key, item in value.items():
            yield from _flatten(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            yield from _flatten(item, f"{prefix}[{index}]")
    else:
        yield prefix, value



def _suite_observation(row: dict) -> dict:
    """Readable suite metrics; large image payloads remain in JSON and CSV."""
    if "metrics" in row:
        return {"status": row.get("status", "completed"), "metrics": row["metrics"],
                **({"reason": row["reason"]} if "reason" in row else {})}
    observation = {}
    for key, value in row.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            if not (isinstance(value, str) and value.startswith("data:image/")):
                observation[key] = value
        elif key in {"feature", "gradient"} and isinstance(value, dict):
            observation[key] = {name: item for name, item in value.items()
                                if isinstance(item, (int, float, bool)) or item is None}
    return observation

def _metric_rows(comparison: dict) -> list[list[str]]:
    rows = [["Metric", "Baseline mean +/- SD", "Candidate mean +/- SD", "Change", "Unit"]]
    for metric, baseline in comparison.get("baseline", {}).items():
        if not isinstance(baseline, dict):
            continue
        candidate = comparison.get("candidate", {}).get(metric, {})
        delta = comparison.get("deltas", {}).get(metric, {})
        def summary(aggregate):
            if aggregate.get("mean") is None:
                return "Unavailable / unbounded"
            return f"{_show(aggregate['mean'])} +/- {_show(aggregate.get('std'))} (n={aggregate.get('n', 0)})"
        rows.append([metric, summary(baseline), summary(candidate), _show(delta.get("absolute")), delta.get("unit", "")])
    return rows


def _markdown(experiment: dict) -> str:
    result = experiment.get("result") or {}
    comparison = result.get("comparison") or {}
    config = experiment.get("config") or {}
    candidate = config.get("candidate", {})
    recommendation = comparison.get("recommendation", {})
    lines = [f"# SFL-Lab Formula Impact Report: {experiment.get('name', config.get('name', 'Experiment'))}", "",
             f"Experiment: `{experiment.get('id', 'unassigned')}` · Status: **{experiment.get('status', 'unknown')}**", "",
             f"Created: {experiment.get('created_at', 'unrecorded')} · Updated: {experiment.get('updated_at', 'unrecorded')}", "",
             "## Research question and candidate", "", config.get("hypothesis", "No hypothesis recorded."), "",
             f"Candidate: **{candidate.get('name', 'unrecorded')}** · Hook: `{candidate.get('hook', 'unrecorded')}` · Mode: {candidate.get('mode', 'unrecorded')}", "",
             "```text", candidate.get("expression", "No expression recorded."), "```", "",
             "## Architecture and methodology", "",
             "Private data -> Client Front -> activation message -> Server Middle -> intermediate message -> Client Back -> prediction -> local label/loss. Backward messages return through both split boundaries; federated aggregation redistributes the model.", "",
             f"Model: {config.get('model', 'unrecorded')}; split: {_show(config.get('split'))}; clients: {config.get('clients', 'unrecorded')}; servers: {config.get('servers', 'unrecorded')}; dataset: {config.get('dataset', 'unrecorded')}; seeds: {_show(config.get('seeds'))}.", "",
             "Checkpoint and parameter selection uses validation performance. Held-out test accuracy is reported separately. Logical locality in a single process does not provide operating-system or cryptographic isolation.", "",
             "## Conclusion", "", f"Hypothesis status: **{comparison.get('hypothesis_status', 'inconclusive')}**", "",
             f"Recommendation: **{recommendation.get('decision', 'INCONCLUSIVE')}**", ""]
    lines += [f"- {reason}" for reason in recommendation.get("reasons", ["No complete comparison is available."])]
    lines += ["", "## Measured performance and modeled costs", ""]
    metric_rows = _metric_rows(comparison)
    for index, row in enumerate(metric_rows):
        lines.append("| " + " | ".join(str(item).replace("|", "\\|").replace("\n", " ") for item in row) + " |")
        if index == 0:
            lines.append("| --- | --- | --- | --- | --- |")
    lines += ["", "Accuracy changes are percentage points. Relative changes use the baseline denominator. Missing/unbounded epsilon is not zero and has no defined percentage reduction. Tensor traffic and runtime are measured; network latency and energy are modeled estimates.", "",
              "## Acceptance evidence", ""]
    for criterion in comparison.get("criteria", []):
        lines.append(f"- **{criterion.get('name')}: {criterion.get('status')}** — observed {_show(criterion.get('observed'))}; threshold {_show(criterion.get('threshold'))}. {criterion.get('detail', '')}")
    lines += ["", "## Statistical evidence", "", "Paired tests match seed identifiers. P-values are two-sided paired t-tests with Holm adjustment over reported metric tests. At least three nonconstant paired observations are required. Acceptance on observed means is distinct from significance.", "",
              "| Metric | Paired seeds | Mean difference | 95% CI | Adjusted p | Effect size (dz) | Status |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for metric, item in comparison.get("statistics", {}).items():
        lines.append("| " + " | ".join([metric, _show(item.get("n_pairs")), _show(item.get("mean_difference")), _show(item.get("ci95")),
                                         _show(item.get("adjusted_p_value")), _show(item.get("effect_size")), item.get("status", "unavailable")]) + " |")
    lines += ["", "## Supplementary evaluations", ""]
    for name, suite in (result.get("suites") or {}).items():
        lines += [f"### {name}", "", f"Status: {suite.get('status', 'unrecorded')}", ""]
        for row in suite.get("rows", []):
            lines.append(f"- **{row.get('label', row.get('scenario', row.get('condition', row.get('name', 'Observation'))))}**: {_show(_suite_observation(row))}")
        if "selected_value" in suite:
            lines.append(f"- Validation-selected {suite.get('parameter')}: {_show(suite.get('selected_value'))}.")
        for limitation in suite.get("limitations", []):
            lines.append(f"- Limitation: {limitation}")
        lines.append("")
    lines += ["## Limitations", ""]
    limitations = list(dict.fromkeys(comparison.get("limitations", []) + result.get("limitations", [])))
    lines += [f"- {item}" for item in limitations] or ["- No limitations were recorded; absence of evidence does not establish generalization."]
    lines += ["", "## Reproducibility configuration", "", "```json", json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False), "```", "",
              "## Environment", "", "```json", json.dumps(result.get("environment", {}), indent=2, ensure_ascii=False, allow_nan=False), "```", "",
              "## Run provenance and privacy assumptions", ""]
    for variant in ("baseline", "candidate"):
        for run in result.get(variant, []):
            provenance = {key: run.get(key) for key in ("seed", "variant", "dataset", "environment", "privacy", "compliance", "stopped_early")}
            lines += [f"### {variant} seed {run.get('seed', 'unrecorded')}", "", "```json", json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False), "```", ""]
    lines += ["## Execution logs", "", "```json", json.dumps(experiment.get("logs", []), indent=2, ensure_ascii=False, allow_nan=False), "```", "",
              "Full raw evidence, learning curves, suite observations, and Pareto coordinates are preserved in the JSON export. The CSV export contains every stored scalar with its evidence path.", ""]
    return "\n".join(lines)


def _pdf(experiment: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.graphics.shapes import Drawing, Line, String
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.graphics.charts.lineplots import LinePlot

    stream = io.BytesIO()
    document = SimpleDocTemplate(stream, pagesize=A4, rightMargin=35, leftMargin=35, topMargin=35, bottomMargin=35)
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 8
    styles["BodyText"].leading = 11
    flow = []
    result, config = experiment.get("result") or {}, experiment.get("config") or {}
    comparison = result.get("comparison") or {}

    def p(value, style="BodyText"):
        text = str(value).replace("ε", "epsilon").replace("±", "+/-").replace("→", "->")
        return Paragraph(escape(text).replace("\n", "<br/>"), styles[style])

    def heading(text):
        flow.extend([Spacer(1, 10), p(text, "Heading2")])

    def table(rows, widths=None):
        wrapped = [[p(cell) for cell in row] for row in rows]
        grid = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
        grid.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17334c")),
                                 ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                 ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#cbd5e1")),
                                 ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                 ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
        flow.append(grid)

    flow += [p("SFL-Lab Formula Impact Report", "Title"), p(experiment.get("name", config.get("name", "Experiment")), "Heading2"),
             p(f"Experiment {experiment.get('id', 'unassigned')} | Status {experiment.get('status', 'unknown')} | {experiment.get('created_at', '')}"), Spacer(1, 12)]
    heading("Hypothesis and candidate")
    flow.append(p(config.get("hypothesis", "No hypothesis recorded.")))
    candidate = config.get("candidate", {})
    flow.append(p(f"Candidate {candidate.get('name', 'unrecorded')} | Hook {candidate.get('hook', 'unrecorded')} | {candidate.get('mode', 'unrecorded')} mode"))
    flow.append(p(candidate.get("expression", "No expression recorded.")))
    heading("U-shaped architecture")
    flow.append(p(f"Private data -> Client Front -> Server Middle -> Client Back -> local prediction/loss. Backward tensor messages return across both cuts; federated aggregation redistributes the model. Split: {_show(config.get('split'))}. Labels and raw inputs remain logically client-local."))
    heading("Conclusion and acceptance evidence")
    recommendation = comparison.get("recommendation", {})
    flow.append(p(f"Hypothesis: {comparison.get('hypothesis_status', 'inconclusive')} | Recommendation: {recommendation.get('decision', 'INCONCLUSIVE')}"))
    for reason in recommendation.get("reasons", []):
        flow.append(p(reason))
    if comparison.get("criteria"):
        table([["Criterion", "Status", "Evidence"]] + [[c.get("name", ""), c.get("status", ""),
               f"Observed {_show(c.get('observed'))}; threshold {_show(c.get('threshold'))}. {c.get('detail', '')}"] for c in comparison["criteria"]], [115, 65, 345])
    heading("Performance and costs")
    table(_metric_rows(comparison), [100, 115, 115, 70, 125])
    flow.append(p("Accuracy differences are percentage points. Runtime and tensor traffic are measured; network latency and energy are modeled estimates. Unavailable/unbounded epsilon has no defined percentage reduction."))
    # Learning curves use only stored observations. No extrapolated points or
    # curves are supplied when actual round histories are absent.
    curves = comparison.get("learning_curves", {})
    for metric, label in (("val_accuracy", "Validation accuracy (%)"), ("val_loss", "Validation loss")):
        data, names = [], []
        for variant in ("baseline", "candidate"):
            points = [(row["round"], row.get(metric, {}).get("mean")) for row in curves.get(variant, [])
                      if row.get(metric, {}).get("mean") is not None]
            if points:
                data.append(points)
                names.append(variant)
        if data:
            heading(label + " by round")
            drawing = Drawing(510, 205)
            chart = LinePlot()
            chart.x, chart.y, chart.width, chart.height = 45, 35, 440, 145
            chart.data = data
            chart.joinedLines = 1
            chart.lines[0].strokeColor = colors.HexColor("#146fa1")
            if len(data) > 1:
                chart.lines[1].strokeColor = colors.HexColor("#d76727")
            x_values = [x for points in data for x, _ in points]
            if min(x_values) == max(x_values):
                chart.xValueAxis.valueMin = min(x_values) - .5
                chart.xValueAxis.valueMax = max(x_values) + .5
            drawing.add(chart)
            drawing.add(String(45, 10, "Round | " + " / ".join(names), fontSize=8))
            flow.append(drawing)
    heading("Statistical reliability")
    table([["Metric", "Pairs", "95% difference CI", "Holm p", "Status"]] + [[metric, item.get("n_pairs"), _show(item.get("ci95")),
           _show(item.get("adjusted_p_value")), item.get("status", "")] for metric, item in comparison.get("statistics", {}).items()], [105, 35, 130, 65, 190])
    flow.append(p("Two-sided paired t-tests match seeds; no inferential test is reported for fewer than three pairs or constant differences. Mean-based acceptance is distinct from significance."))
    heading("Supplementary studies")
    for name, suite in (result.get("suites") or {}).items():
        flow.append(p(f"{name}: {suite.get('status', 'unrecorded')}", "Heading3"))
        for row in suite.get("rows", []):
            flow.append(p(f"{row.get('label', row.get('scenario', row.get('condition', row.get('name', 'Observation'))))}: {_show(_suite_observation(row))}"))
        for limitation in suite.get("limitations", []):
            flow.append(p("Limitation: " + limitation))
    heading("Limitations")
    for limitation in dict.fromkeys(comparison.get("limitations", []) + result.get("limitations", [])):
        flow.append(p(limitation))
    heading("Configuration and reproducibility")
    for path, value in _flatten(config):
        flow.append(p(f"{path}: {_show(value)}"))
    heading("Environment")
    for path, value in _flatten(result.get("environment", {})):
        flow.append(p(f"{path}: {_show(value)}"))
    for variant in ("baseline", "candidate"):
        for run in result.get(variant, []):
            flow.append(p(f"{variant} seed {run.get('seed', 'unrecorded')}", "Heading3"))
            for key in ("dataset", "privacy", "environment", "compliance"):
                for path, value in _flatten(run.get(key, {}), key):
                    flow.append(p(f"{path}: {_show(value)}"))
    heading("Execution log")
    for entry in experiment.get("logs", []):
        flow.append(p(_show(entry)))
    flow.append(p("Download JSON for all raw measurements, per-round histories, suite observations, attack reconstructions, and complete metadata. CSV preserves every scalar with its evidence path."))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(35, 20, f"SFL-Lab | {experiment.get('id', 'unassigned')} | Page {doc.page}")
        canvas.restoreState()
    document.build(flow, onFirstPage=footer, onLaterPages=footer)
    return stream.getvalue()


def make_report(experiment: dict, format: str) -> bytes:
    """Return an evidence-preserving export; callers choose download headers."""
    record = _safe(experiment)
    if format == "json":
        return json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    if format == "csv":
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(["evidence_path", "value", "value_type", "variant"])
        for path, value in _flatten(record):
            text = json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (list, dict)) else "" if value is None else str(value)
            # Spreadsheet exports should display user input rather than execute
            # it as a formula. JSON retains the exact original text.
            if isinstance(value, str) and text.startswith(("=", "+", "-", "@", "\t", "\r")):
                text = "'" + text
            variant = "baseline" if path.startswith("result.baseline[") else "candidate" if path.startswith("result.candidate[") else ""
            writer.writerow([path, text, type(value).__name__, variant])
        return stream.getvalue().encode("utf-8-sig")
    if format in {"markdown", "md"}:
        return _markdown(record).encode("utf-8")
    if format == "pdf":
        return _pdf(record)
    raise ValueError("Report format must be json, csv, pdf, or markdown")
