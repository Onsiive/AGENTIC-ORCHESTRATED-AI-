"""
comparison_report.py
────────────────────
Generates a side-by-side comparison report between two Agentic AI systems
(e.g. Qwen local vs Gemini cloud) after both workflows have completed.

Outputs:
  • Console print (human-readable, aligned columns)
  • JSON report   → logs/comparison_report_<timestamp>.json
  • CSV summary   → logs/comparison_summary_<timestamp>.csv
"""

import json
import csv
import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CONSOLE PRINTER
# ──────────────────────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float = 100.0, width: int = 20) -> str:
    """Simple ASCII progress bar."""
    if max_val == 0:
        filled = 0
    else:
        filled = int(round(value / max_val * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _pct(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:.2f}%"


def _num(v) -> str:
    if v is None:
        return "N/A"
    return str(v)


def print_comparison_report(
    agent_a_name: str,
    agent_b_name: str,
    metrics_a: dict,
    metrics_b: dict,
):
    """Print a formatted side-by-side comparison to console."""

    W = 74
    C1 = 28   # label column width
    C2 = 20   # agent A column
    C3 = 20   # agent B column

    def divider(char="─"):
        print(char * W)

    def row(label, val_a, val_b, unit=""):
        la = f"{val_a}{unit}" if val_a != "N/A" else "N/A"
        lb = f"{val_b}{unit}" if val_b != "N/A" else "N/A"
        print(f"  {label:<{C1}} {la:>{C2}} {lb:>{C3}}")

    def section(title):
        print()
        divider()
        print(f"  {title}")
        divider()

    def header_row():
        print(f"  {'Metric':<{C1}} {agent_a_name:>{C2}} {agent_b_name:>{C3}}")
        divider("─")

    # ── Title ──────────────────────────────────────────────────────────────
    print()
    print("═" * W)
    print("  🤖  AGENTIC AI COMPARISON REPORT — SIDE BY SIDE")
    print("═" * W)
    print(f"  System A : {agent_a_name}")
    print(f"  System B : {agent_b_name}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    divider("═")

    # ── Model Performance ─────────────────────────────────────────────────
    section("📊  1. MODEL PERFORMANCE (Conversion Prediction)")
    header_row()

    metrics = [
        ("Accuracy",  "accuracy",  "%"),
        ("Precision", "precision", "%"),
        ("Recall",    "recall",    "%"),
        ("F1-Score",  "f1_score",  "%"),
        ("ROC-AUC",   "roc_auc",   "%"),
        ("Model Used","model_used",""),
    ]
    for label, key, unit in metrics:
        va = metrics_a.get(key, "N/A")
        vb = metrics_b.get(key, "N/A")
        if isinstance(va, float):
            va = f"{va:.2f}"
        if isinstance(vb, float):
            vb = f"{vb:.2f}"
        row(label, va, vb, unit)

    # winner
    try:
        acc_a = float(metrics_a.get('accuracy', 0))
        acc_b = float(metrics_b.get('accuracy', 0))
        if acc_a > acc_b:
            winner = f"✅ {agent_a_name} unggul pada akurasi"
        elif acc_b > acc_a:
            winner = f"✅ {agent_b_name} unggul pada akurasi"
        else:
            winner = "🤝 Kedua sistem memiliki akurasi yang sama"
        print(f"\n  → {winner}")
    except Exception:
        pass

    # ── LLM Orchestrator Reliability ─────────────────────────────────────
    section("🧠  2. LLM ORCHESTRATOR RELIABILITY")
    header_row()

    rel_metrics = [
        ("Total LLM Calls",    "llm_call_total",    ""),
        ("Successful Calls",   "llm_call_success",  ""),
        ("Retry Attempts",     "llm_call_retry",    ""),
        ("Fallback Used",      "fallback_count",    "x"),
        ("Reliability Score",  "llm_reliability_pct", "%"),
        ("Fallback Rate",      "fallback_rate_pct", "%"),
    ]
    for label, key, unit in rel_metrics:
        va = metrics_a.get(key, "N/A")
        vb = metrics_b.get(key, "N/A")
        if isinstance(va, float):
            va = f"{va:.2f}"
        if isinstance(vb, float):
            vb = f"{vb:.2f}"
        row(label, va, vb, unit)

    # Reliability bar chart
    rel_a = metrics_a.get('llm_reliability_pct', 0) or 0
    rel_b = metrics_b.get('llm_reliability_pct', 0) or 0
    print(f"\n  Reliability Visual:")
    print(f"  {agent_a_name:<20} [{_bar(rel_a)}] {rel_a:.1f}%")
    print(f"  {agent_b_name:<20} [{_bar(rel_b)}] {rel_b:.1f}%")

    try:
        if rel_a > rel_b:
            winner = f"✅ {agent_a_name} lebih andal (lebih sedikit retry/fallback)"
        elif rel_b > rel_a:
            winner = f"✅ {agent_b_name} lebih andal (lebih sedikit retry/fallback)"
        else:
            winner = "🤝 Keandalan orchestrator sama"
        print(f"\n  → {winner}")
    except Exception:
        pass

    # ── Workflow Execution ────────────────────────────────────────────────
    section("⏱️   3. WORKFLOW EXECUTION")
    header_row()

    exec_metrics = [
        ("Duration (seconds)",     "duration_seconds",       "s"),
        ("Total Steps",            "total_steps",            ""),
        ("Successful Steps",       "success_steps",          ""),
        ("Task Completion Rate",   "task_completion_rate",   "%"),
    ]
    for label, key, unit in exec_metrics:
        va = metrics_a.get(key, "N/A")
        vb = metrics_b.get(key, "N/A")
        if isinstance(va, float):
            va = f"{va:.2f}"
        if isinstance(vb, float):
            vb = f"{vb:.2f}"
        row(label, va, vb, unit)

    try:
        dur_a = metrics_a.get('duration_seconds') or float('inf')
        dur_b = metrics_b.get('duration_seconds') or float('inf')
        if dur_a < dur_b:
            diff = dur_b - dur_a
            winner = f"✅ {agent_a_name} lebih cepat ({diff:.1f}s)"
        elif dur_b < dur_a:
            diff = dur_a - dur_b
            winner = f"✅ {agent_b_name} lebih cepat ({diff:.1f}s)"
        else:
            winner = "🤝 Durasi sama"
        print(f"\n  → {winner}")
    except Exception:
        pass

    # ── Recommendations ───────────────────────────────────────────────────
    section("📋  4. RECOMMENDATION OUTPUT")
    header_row()

    row(
        "Total Recommendations",
        _num(metrics_a.get('total_recommendations')),
        _num(metrics_b.get('total_recommendations')),
    )

    # Priority breakdown
    print(f"\n  Priority Distribution:")
    for prio in ['Critical', 'High', 'Growth', 'Advisory']:
        va = metrics_a.get('rec_by_priority', {}).get(prio, 0)
        vb = metrics_b.get('rec_by_priority', {}).get(prio, 0)
        if va > 0 or vb > 0:
            row(f"  └─ {prio}", _num(va), _num(vb))

    # Decision area breakdown
    print(f"\n  Decision Area Breakdown:")
    all_areas = set(
        list(metrics_a.get('rec_by_area', {}).keys()) +
        list(metrics_b.get('rec_by_area', {}).keys())
    )
    for area in sorted(all_areas):
        va = metrics_a.get('rec_by_area', {}).get(area, 0)
        vb = metrics_b.get('rec_by_area', {}).get(area, 0)
        label = area[:26] + ".." if len(area) > 26 else area
        row(f"  └─ {label}", _num(va), _num(vb))

    # ── Agentic AI Evaluation ─────────────────────────────────────────────
    section("🧭  5. AGENTIC AI EVALUATION FRAMEWORK")
    header_row()

    agentic_metrics = [
        ("Overall Agentic Score", "agentic_overall_score", "/100"),
        ("Planning Quality", "planning_score", "/100"),
        ("Tool Selection", "tool_selection_score", "/100"),
        ("Reasoning Trace", "reasoning_trace_score", "/100"),
        ("Robustness", "robustness_score", "/100"),
        ("Governance & Safety", "governance_safety_score", "/100"),
        ("Benchmark Coverage", "benchmark_score", "/100"),
        ("Multi-Agent Coordination", "multi_agent_score", "/100"),
        ("Autonomy-Control Balance", "autonomy_control_score", "/100"),
    ]
    for label, key, unit in agentic_metrics:
        va = metrics_a.get(key, "N/A")
        vb = metrics_b.get(key, "N/A")
        if isinstance(va, float):
            va = f"{va:.2f}"
        if isinstance(vb, float):
            vb = f"{vb:.2f}"
        row(label, va, vb, unit)

    overall_a = metrics_a.get('agentic_overall_score', 0) or 0
    overall_b = metrics_b.get('agentic_overall_score', 0) or 0
    print(f"\n  Agentic Score Visual:")
    print(f"  {agent_a_name:<20} [{_bar(overall_a)}] {overall_a:.1f}/100")
    print(f"  {agent_b_name:<20} [{_bar(overall_b)}] {overall_b:.1f}/100")

    path_a = (metrics_a.get('agentic_report_paths') or {}).get('json')
    path_b = (metrics_b.get('agentic_report_paths') or {}).get('json')
    if path_a or path_b:
        print("\n  Agentic evaluation artifacts:")
        if path_a:
            print(f"  • {agent_a_name}: {path_a}")
        if path_b:
            print(f"  • {agent_b_name}: {path_b}")

    # ── Final Summary ─────────────────────────────────────────────────────
    section("🏆  OVERALL COMPARISON SUMMARY")

    scores = {"a": 0, "b": 0}

    def compare(key, higher_is_better=True):
        va = metrics_a.get(key)
        vb = metrics_b.get(key)
        if va is None or vb is None:
            return
        try:
            va, vb = float(va), float(vb)
            if higher_is_better:
                if va > vb: scores["a"] += 1
                elif vb > va: scores["b"] += 1
            else:
                if va < vb: scores["a"] += 1
                elif vb < va: scores["b"] += 1
        except Exception:
            pass

    compare("accuracy")
    compare("f1_score")
    compare("roc_auc")
    compare("llm_reliability_pct")
    compare("fallback_rate_pct", higher_is_better=False)
    compare("duration_seconds", higher_is_better=False)
    compare("agentic_overall_score")
    compare("planning_score")
    compare("tool_selection_score")
    compare("reasoning_trace_score")
    compare("robustness_score")
    compare("governance_safety_score")

    print(f"  {agent_a_name:<30} Skor: {scores['a']} / 12 kriteria")
    print(f"  {agent_b_name:<30} Skor: {scores['b']} / 12 kriteria")
    print()

    if scores["a"] > scores["b"]:
        verdict = f"🥇 {agent_a_name} unggul secara keseluruhan"
    elif scores["b"] > scores["a"]:
        verdict = f"🥇 {agent_b_name} unggul secara keseluruhan"
    else:
        verdict = "🤝 Kedua sistem setara — unggul di kriteria berbeda"

    print(f"  Kesimpulan: {verdict}")
    divider("═")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# FILE SAVERS
# ──────────────────────────────────────────────────────────────────────────────

def save_comparison_report(
    agent_a_name: str,
    agent_b_name: str,
    metrics_a: dict,
    metrics_b: dict,
    output_dir: str = "logs",
) -> dict:
    """
    Save comparison results to JSON and CSV.
    Returns dict with paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── JSON ──────────────────────────────────────────────────────────────
    json_path = os.path.join(
        output_dir, f"comparison_report_{ts}.json"
    )
    report = {
        "generated_at": datetime.now().isoformat(),
        "agent_a": {"name": agent_a_name, "metrics": metrics_a},
        "agent_b": {"name": agent_b_name, "metrics": metrics_b},
        "comparison": _build_comparison_dict(
            agent_a_name, agent_b_name, metrics_a, metrics_b
        ),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"📄 Comparison JSON saved: {json_path}")

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(
        output_dir, f"comparison_summary_{ts}.csv"
    )
    flat_rows = _flatten_for_csv(agent_a_name, agent_b_name, metrics_a, metrics_b)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["metric", agent_a_name, agent_b_name, "winner"]
        )
        writer.writeheader()
        writer.writerows(flat_rows)
    logger.info(f"📊 Comparison CSV saved: {csv_path}")

    return {"json": json_path, "csv": csv_path}


def _build_comparison_dict(name_a, name_b, m_a, m_b):
    """Build winner analysis dict."""
    result = {}
    comparable = [
        ("accuracy",           True),
        ("precision",          True),
        ("recall",             True),
        ("f1_score",           True),
        ("roc_auc",            True),
        ("llm_reliability_pct",True),
        ("fallback_rate_pct",  False),
        ("duration_seconds",   False),
        ("total_recommendations", True),
        ("agentic_overall_score", True),
        ("planning_score", True),
        ("tool_selection_score", True),
        ("reasoning_trace_score", True),
        ("robustness_score", True),
        ("governance_safety_score", True),
        ("benchmark_score", True),
        ("multi_agent_score", True),
        ("autonomy_control_score", True),
    ]
    for key, higher_better in comparable:
        va = m_a.get(key)
        vb = m_b.get(key)
        try:
            va_f, vb_f = float(va), float(vb)
            if higher_better:
                winner = name_a if va_f > vb_f else (name_b if vb_f > va_f else "tie")
            else:
                winner = name_a if va_f < vb_f else (name_b if vb_f < va_f else "tie")
            diff = round(va_f - vb_f, 4)
        except Exception:
            winner = "N/A"
            diff = None
        result[key] = {
            name_a: va, name_b: vb,
            "winner": winner, "difference": diff,
        }
    return result


def _flatten_for_csv(name_a, name_b, m_a, m_b):
    """Flatten metrics dict into rows for CSV."""
    metric_labels = [
        ("accuracy",              "Accuracy (%)"),
        ("precision",             "Precision (%)"),
        ("recall",                "Recall (%)"),
        ("f1_score",              "F1-Score (%)"),
        ("roc_auc",               "ROC-AUC (%)"),
        ("model_used",            "Model Used"),
        ("duration_seconds",      "Duration (seconds)"),
        ("total_steps",           "Total Steps"),
        ("success_steps",         "Successful Steps"),
        ("task_completion_rate",  "Task Completion Rate (%)"),
        ("llm_call_total",        "Total LLM Calls"),
        ("llm_call_success",      "Successful LLM Calls"),
        ("llm_call_retry",        "LLM Retry Attempts"),
        ("fallback_count",        "Fallback Used (count)"),
        ("llm_reliability_pct",   "LLM Reliability (%)"),
        ("fallback_rate_pct",     "Fallback Rate (%)"),
        ("total_recommendations", "Total Recommendations"),
        ("agentic_overall_score", "Overall Agentic Score"),
        ("planning_score", "Planning Quality"),
        ("tool_selection_score", "Tool Selection"),
        ("reasoning_trace_score", "Reasoning Trace"),
        ("robustness_score", "Robustness"),
        ("governance_safety_score", "Governance & Safety"),
        ("benchmark_score", "Benchmark Coverage"),
        ("multi_agent_score", "Multi-Agent Coordination"),
        ("autonomy_control_score", "Autonomy-Control Balance"),
    ]
    rows = []
    higher_better = {
        "accuracy", "precision", "recall", "f1_score", "roc_auc",
        "success_steps", "task_completion_rate",
        "llm_call_success", "llm_reliability_pct", "total_recommendations",
        "agentic_overall_score", "planning_score", "tool_selection_score",
        "reasoning_trace_score", "robustness_score", "governance_safety_score",
        "benchmark_score", "multi_agent_score", "autonomy_control_score",
    }
    lower_better = {"duration_seconds", "llm_call_retry", "fallback_count", "fallback_rate_pct"}

    for key, label in metric_labels:
        va = m_a.get(key, "")
        vb = m_b.get(key, "")
        winner = ""
        try:
            va_f, vb_f = float(va), float(vb)
            if key in higher_better:
                if va_f > vb_f: winner = name_a
                elif vb_f > va_f: winner = name_b
                else: winner = "Tie"
            elif key in lower_better:
                if va_f < vb_f: winner = name_a
                elif vb_f < va_f: winner = name_b
                else: winner = "Tie"
        except Exception:
            pass
        rows.append({"metric": label, name_a: va, name_b: vb, "winner": winner})
    return rows
