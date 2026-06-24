"""
Agentic AI evaluation utilities for the Digital Marketing decision workflow.

This module upgrades the project from a normal ML pipeline into an auditable
agentic AI evaluation framework.  It evaluates the agent trajectory, planning,
tool selection, collaboration, robustness, governance, safety, and benchmark
coverage without removing the original analytics/recommendation features.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_WORKFLOW_TOOLS = [
    "load_and_inspect_data",
    "preprocess_data",
    "analyze_data",
    "generate_recommendations",
]

AGENT_ROLE_BY_TOOL = {
    "load_and_inspect_data": "perception_agent",
    "preprocess_data": "preprocessing_agent",
    "analyze_data": "analysis_agent",
    "generate_recommendations": "optimization_recommendation_agent",
}

TOOL_PRECONDITIONS = {
    "load_and_inspect_data": [],
    "preprocess_data": ["load_and_inspect_data"],
    "analyze_data": ["preprocess_data"],
    "generate_recommendations": ["analyze_data"],
}


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.upper() == "N/A":
            return default
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def _pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(max(0.0, min(1.0, numerator / denominator)) * 100, 2)


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion for pandas/numpy/sklearn objects."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.head(50).to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.head(50).to_list()
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


# ---------------------------------------------------------------------------
# Trajectory logging
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryEvent:
    timestamp: str
    step_number: int
    agent_role: str
    thought: str
    action: str
    observation: str
    reflection: str
    success: bool
    duration_seconds: Optional[float] = None
    decision_source: str = "llm"
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgenticTrajectoryLogger:
    """
    Stores thought/action/observation/reflection records.

    The `thought` field is a short, user-visible rationale produced by the
    planner prompt. It is intentionally not a hidden chain-of-thought dump.
    """

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(
            self.output_dir,
            f"agentic_trajectory_{self.started_at}.jsonl",
        )
        self.events: List[TrajectoryEvent] = []

    def log_event(
        self,
        step_number: int,
        action: str,
        observation: str,
        success: bool,
        thought: str = "",
        reflection: str = "",
        duration_seconds: Optional[float] = None,
        decision_source: str = "llm",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event = TrajectoryEvent(
            timestamp=datetime.now().isoformat(),
            step_number=step_number,
            agent_role=AGENT_ROLE_BY_TOOL.get(action, "planner_agent"),
            thought=thought or "Planner selected the next workflow action.",
            action=action,
            observation=observation,
            reflection=reflection or self._default_reflection(success, action),
            success=bool(success),
            duration_seconds=duration_seconds,
            decision_source=decision_source,
            metadata=metadata or {},
        )
        self.events.append(event)
        row = _jsonable(asdict(event))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def _default_reflection(self, success: bool, action: str) -> str:
        if success:
            return f"The {action} action completed and the workflow can continue."
        return f"The {action} action failed; the adaptive workflow should recover or request HITL."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_path": self.path,
            "event_count": len(self.events),
            "events": [_jsonable(asdict(e)) for e in self.events],
        }


# ---------------------------------------------------------------------------
# Planning and tool-use evaluation
# ---------------------------------------------------------------------------

class PlanningEvaluator:
    def evaluate(self, completed_steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        tool_sequence = [s.get("tool") for s in completed_steps if s.get("tool")]
        successful_tools = [s.get("tool") for s in completed_steps if s.get("success")]

        coverage = _pct(
            len(set(successful_tools).intersection(REQUIRED_WORKFLOW_TOOLS)),
            len(REQUIRED_WORKFLOW_TOOLS),
        )
        order_score = self._order_score(tool_sequence)
        redundancy_penalty = self._redundancy_penalty(tool_sequence)
        failure_penalty = self._failure_penalty(completed_steps)

        score = round(max(0.0, coverage * 0.45 + order_score * 0.35 + (100 - redundancy_penalty) * 0.10 + (100 - failure_penalty) * 0.10), 2)
        return {
            "score": score,
            "required_tool_coverage_pct": coverage,
            "tool_order_score_pct": order_score,
            "redundancy_penalty_pct": redundancy_penalty,
            "failure_penalty_pct": failure_penalty,
            "observed_plan": tool_sequence,
            "expected_plan": REQUIRED_WORKFLOW_TOOLS,
            "interpretation": self._interpret(score),
        }

    def _order_score(self, sequence: Sequence[str]) -> float:
        if not sequence:
            return 0.0
        expected_positions = {tool: i for i, tool in enumerate(REQUIRED_WORKFLOW_TOOLS)}
        violations = 0
        comparable = 0
        seen_expected = []
        for tool in sequence:
            if tool in expected_positions:
                seen_expected.append(tool)
        for i in range(len(seen_expected)):
            for j in range(i + 1, len(seen_expected)):
                comparable += 1
                if expected_positions[seen_expected[i]] > expected_positions[seen_expected[j]]:
                    violations += 1
        if comparable == 0:
            return 100.0 if seen_expected else 0.0
        return round((1 - violations / comparable) * 100, 2)

    def _redundancy_penalty(self, sequence: Sequence[str]) -> float:
        if not sequence:
            return 0.0
        duplicate_count = len(sequence) - len(set(sequence))
        return _pct(duplicate_count, max(1, len(sequence)))

    def _failure_penalty(self, completed_steps: Sequence[Dict[str, Any]]) -> float:
        if not completed_steps:
            return 0.0
        failures = sum(1 for s in completed_steps if not s.get("success"))
        return _pct(failures, len(completed_steps))

    def _interpret(self, score: float) -> str:
        if score >= 85:
            return "Strong planning: tool sequence is complete, ordered, and efficient."
        if score >= 70:
            return "Adequate planning with minor inefficiency or recovery events."
        if score >= 50:
            return "Partial planning quality; workflow should improve sequencing/recovery."
        return "Weak planning quality; workflow resembles brittle orchestration."


class ToolSelectionEvaluator:
    def evaluate(self, completed_steps: Sequence[Dict[str, Any]], invalid_decisions: int = 0) -> Dict[str, Any]:
        successful_so_far = set()
        precondition_checks = []
        valid_actions = 0
        total_actions = 0

        for step in completed_steps:
            tool = step.get("tool")
            if not tool:
                continue
            total_actions += 1
            is_known = tool in REQUIRED_WORKFLOW_TOOLS
            if is_known:
                valid_actions += 1
            required = TOOL_PRECONDITIONS.get(tool, [])
            satisfied = all(req in successful_so_far for req in required)
            precondition_checks.append({
                "tool": tool,
                "required_before": required,
                "preconditions_satisfied": bool(satisfied),
            })
            if step.get("success"):
                successful_so_far.add(tool)

        validity = _pct(valid_actions, total_actions or 1)
        precondition_score = _pct(
            sum(1 for c in precondition_checks if c["preconditions_satisfied"]),
            len(precondition_checks) or 1,
        )
        invalid_penalty = min(100.0, invalid_decisions * 15.0)
        score = round(max(0.0, validity * 0.45 + precondition_score * 0.45 + (100 - invalid_penalty) * 0.10), 2)

        return {
            "score": score,
            "tool_validity_pct": validity,
            "precondition_satisfaction_pct": precondition_score,
            "invalid_decision_count": invalid_decisions,
            "precondition_checks": precondition_checks,
            "interpretation": "Tool selection is context-aware." if score >= 80 else "Tool selection needs stronger state/precondition awareness.",
        }


# ---------------------------------------------------------------------------
# Robustness, safety, benchmark and multi-agent evaluation
# ---------------------------------------------------------------------------

class RobustnessTester:
    """Lightweight robustness suite for fitted model + digital marketing data."""

    def run(
        self,
        raw_data: Optional[pd.DataFrame],
        preprocessed_data: Optional[pd.DataFrame],
        analysis_results: Optional[Dict[str, Any]],
        task: str,
    ) -> Dict[str, Any]:
        report = {
            "score": 0.0,
            "tests": [],
            "data_quality": {},
            "interpretation": "Robustness was not evaluated.",
        }

        data_quality = self._data_quality(raw_data, preprocessed_data)
        report["data_quality"] = data_quality

        if not analysis_results:
            report["interpretation"] = "Analysis results unavailable, so only data quality checks were performed."
            report["score"] = data_quality.get("score", 0.0)
            return report

        evaluation = analysis_results.get("evaluation", {}) if isinstance(analysis_results, dict) else {}
        model = evaluation.get("fitted_model")
        X_test = evaluation.get("X_test")
        y_pred = evaluation.get("predictions")

        perturbation_scores = []
        if model is not None and isinstance(X_test, pd.DataFrame) and y_pred is not None:
            perturbation_scores.extend(self._prediction_stability_tests(model, X_test, np.asarray(y_pred), task))
        else:
            report["tests"].append({
                "name": "prediction_stability",
                "status": "skipped",
                "reason": "No fitted model or X_test available from analysis agent.",
                "score": 0.0,
            })

        report["tests"].extend(perturbation_scores)
        test_scores = [t.get("score", 0.0) for t in report["tests"] if t.get("status") != "skipped"]
        if test_scores:
            report["score"] = round(statistics.mean(test_scores) * 0.70 + data_quality.get("score", 0.0) * 0.30, 2)
        else:
            report["score"] = data_quality.get("score", 0.0)
        report["interpretation"] = self._interpret(report["score"])
        return report

    def _data_quality(self, raw_data: Optional[pd.DataFrame], preprocessed_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        df = raw_data if raw_data is not None else preprocessed_data
        if df is None or getattr(df, "empty", True):
            return {"score": 0.0, "status": "unavailable"}

        missing_ratio = float(df.isna().sum().sum() / max(1, df.shape[0] * df.shape[1]))
        duplicate_ratio = float(df.duplicated().sum() / max(1, len(df)))
        numeric = df.select_dtypes(include=[np.number])
        outlier_ratio = 0.0
        if not numeric.empty:
            z = np.abs((numeric - numeric.mean(numeric_only=True)) / (numeric.std(numeric_only=True).replace(0, np.nan)))
            outlier_ratio = float((z > 4).sum().sum() / max(1, numeric.shape[0] * numeric.shape[1]))
        penalty = min(100.0, missing_ratio * 80 + duplicate_ratio * 40 + outlier_ratio * 60)
        return {
            "score": round(max(0.0, 100 - penalty), 2),
            "missing_ratio": round(missing_ratio, 4),
            "duplicate_ratio": round(duplicate_ratio, 4),
            "extreme_outlier_ratio": round(outlier_ratio, 4),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        }

    def _prediction_stability_tests(self, model: Any, X_test: pd.DataFrame, baseline_pred: np.ndarray, task: str) -> List[Dict[str, Any]]:
        tests = []
        sample = X_test.copy().head(min(500, len(X_test)))
        baseline = np.asarray(model.predict(sample))
        numeric_cols = list(sample.select_dtypes(include=[np.number]).columns)

        # 1. Row-order invariance.
        try:
            shuffled = sample.sample(frac=1.0, random_state=7)
            shuffled_pred = pd.Series(model.predict(shuffled), index=shuffled.index).loc[sample.index].to_numpy()
            tests.append(self._stability_result("row_order_invariance", baseline, shuffled_pred, task))
        except Exception as exc:
            tests.append({"name": "row_order_invariance", "status": "failed", "reason": str(exc), "score": 0.0})

        # 2. Small numeric noise.
        if numeric_cols:
            try:
                noisy = sample.copy()
                rng = np.random.default_rng(42)
                for col in numeric_cols:
                    std = float(noisy[col].std() or 0.0)
                    if std > 0:
                        noisy[col] = noisy[col] + rng.normal(0, std * 0.01, size=len(noisy))
                noisy_pred = np.asarray(model.predict(noisy))
                tests.append(self._stability_result("small_numeric_noise", baseline, noisy_pred, task))
            except Exception as exc:
                tests.append({"name": "small_numeric_noise", "status": "failed", "reason": str(exc), "score": 0.0})

            # 3. Missing-value recovery by median imputation.
            try:
                missing = sample.copy()
                rng = np.random.default_rng(17)
                mask = rng.random((len(missing), len(numeric_cols))) < 0.03
                for j, col in enumerate(numeric_cols):
                    missing.loc[mask[:, j], col] = np.nan
                    missing[col] = missing[col].fillna(sample[col].median())
                missing_pred = np.asarray(model.predict(missing))
                tests.append(self._stability_result("missing_value_recovery", baseline, missing_pred, task))
            except Exception as exc:
                tests.append({"name": "missing_value_recovery", "status": "failed", "reason": str(exc), "score": 0.0})
        else:
            tests.append({"name": "numeric_perturbations", "status": "skipped", "reason": "No numeric columns in X_test.", "score": 0.0})

        return tests

    def _stability_result(self, name: str, baseline: np.ndarray, candidate: np.ndarray, task: str) -> Dict[str, Any]:
        if task == "regression":
            denom = float(np.std(baseline) or np.mean(np.abs(baseline)) or 1.0)
            normalized_error = float(np.mean(np.abs(baseline - candidate)) / denom)
            stability = max(0.0, min(1.0, 1.0 - normalized_error))
        else:
            stability = float(np.mean(baseline == candidate)) if len(baseline) else 0.0
        return {
            "name": name,
            "status": "passed" if stability >= 0.80 else "warning",
            "stability_pct": round(stability * 100, 2),
            "score": round(stability * 100, 2),
        }

    def _interpret(self, score: float) -> str:
        if score >= 85:
            return "Model and data workflow are stable under lightweight perturbation tests."
        if score >= 70:
            return "Robustness is acceptable but should be strengthened for publication-grade experiments."
        if score >= 50:
            return "Robustness is moderate; add more stress tests before claiming deployment readiness."
        return "Robustness is weak or insufficiently measured."


class GovernanceSafetyEvaluator:
    def evaluate(
        self,
        recommendations: Any,
        hitl_events: Sequence[Dict[str, Any]],
        audit_log_path: Optional[str],
        trajectory_path: Optional[str],
        dataset_columns: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        risks = []
        controls = []

        if hitl_events:
            controls.append("HITL review recorded")
        else:
            risks.append("No HITL decision was recorded for recommendations.")

        if audit_log_path and os.path.exists(audit_log_path):
            controls.append("HITL audit log exists")
        else:
            risks.append("HITL audit log was not found.")

        if trajectory_path and os.path.exists(trajectory_path):
            controls.append("Agent trajectory log exists")
        else:
            risks.append("Agent trajectory log was not found.")

        pii_columns = self._detect_pii_columns(dataset_columns or [])
        if pii_columns:
            risks.append(f"Potential PII-like columns detected: {pii_columns}")
        else:
            controls.append("No obvious PII-like column names detected")

        rec_risks = self._recommendation_risks(recommendations)
        risks.extend(rec_risks)
        if not rec_risks:
            controls.append("Recommendations passed basic safety wording checks")

        base = 100.0
        penalty = min(80.0, len(risks) * 12.0)
        score = round(max(0.0, base - penalty), 2)
        return {
            "score": score,
            "risk_count": len(risks),
            "control_count": len(controls),
            "risks": risks,
            "controls": controls,
            "interpretation": "Governance controls are visible and auditable." if score >= 80 else "Governance evidence needs strengthening.",
        }

    def _detect_pii_columns(self, columns: Sequence[str]) -> List[str]:
        pii_terms = ["email", "phone", "address", "name", "nik", "ktp", "birth", "dob"]
        found = []
        for col in columns:
            low = str(col).lower()
            if any(term in low for term in pii_terms):
                found.append(str(col))
        return found

    def _recommendation_risks(self, recommendations: Any) -> List[str]:
        if recommendations is None:
            return ["No recommendation artifact available for safety review."]
        if isinstance(recommendations, pd.DataFrame):
            if recommendations.empty:
                return []
            text = " ".join(recommendations.astype(str).head(100).fillna("").values.ravel()).lower()
        else:
            text = str(recommendations).lower()
        risky_terms = [
            "guarantee profit", "manipulate", "deceive", "mislead", "discriminate",
            "sensitive attribute", "illegal", "fake review", "dark pattern",
        ]
        return [f"Recommendation text contains risky phrase: '{term}'" for term in risky_terms if term in text]


class MultiAgentEvaluator:
    def evaluate(self, completed_steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        roles = []
        for step in completed_steps:
            role = AGENT_ROLE_BY_TOOL.get(step.get("tool"))
            if role and step.get("success"):
                roles.append(role)
        unique_roles = sorted(set(roles))
        role_coverage = _pct(len(unique_roles), len(set(AGENT_ROLE_BY_TOOL.values())))
        handoff_count = max(0, len(roles) - 1)
        score = round(role_coverage * 0.75 + min(100.0, handoff_count / 3 * 100) * 0.25, 2)
        return {
            "score": score,
            "roles_executed": unique_roles,
            "role_coverage_pct": role_coverage,
            "handoff_count": handoff_count,
            "interpretation": "Multi-agent workflow is explicit and traceable." if score >= 80 else "Agent roles/handoffs should be made more explicit.",
        }


class BenchmarkSystem:
    def evaluate(
        self,
        performance_metrics: Dict[str, Any],
        completed_steps: Sequence[Dict[str, Any]],
        robustness_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        benchmark_table = performance_metrics.get("benchmark_table") or []
        model_count = len(benchmark_table)
        successful_models = sum(1 for row in benchmark_table if "OK" in str(row.get("status", "")))
        model_benchmark_score = _pct(successful_models, model_count or 1) if model_count else 0.0
        workflow_coverage_score = _pct(
            len(set(s.get("tool") for s in completed_steps if s.get("success")).intersection(REQUIRED_WORKFLOW_TOOLS)),
            len(REQUIRED_WORKFLOW_TOOLS),
        )
        robustness_score = _safe_float(robustness_report.get("score"), 0.0)
        score = round(model_benchmark_score * 0.40 + workflow_coverage_score * 0.35 + robustness_score * 0.25, 2)
        return {
            "score": score,
            "model_benchmark_count": model_count,
            "successful_model_benchmarks": successful_models,
            "model_benchmark_score_pct": model_benchmark_score,
            "workflow_benchmark_coverage_pct": workflow_coverage_score,
            "robustness_score_pct": robustness_score,
            "interpretation": "Benchmark coverage supports agentic AI evaluation claims." if score >= 75 else "Benchmark system needs more scenarios and baselines.",
        }


class AgenticEvaluator:
    def __init__(self):
        self.planning = PlanningEvaluator()
        self.tool_selection = ToolSelectionEvaluator()
        self.multi_agent = MultiAgentEvaluator()
        self.benchmark = BenchmarkSystem()

    def evaluate(
        self,
        goal: str,
        completed_steps: Sequence[Dict[str, Any]],
        trajectory: Dict[str, Any],
        llm_stats: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        robustness_report: Dict[str, Any],
        governance_report: Dict[str, Any],
        invalid_decisions: int = 0,
    ) -> Dict[str, Any]:
        planning_report = self.planning.evaluate(completed_steps)
        tool_report = self.tool_selection.evaluate(completed_steps, invalid_decisions=invalid_decisions)
        trace_report = self._trace_evaluation(trajectory)
        autonomy_report = self._autonomy_evaluation(llm_stats, completed_steps)
        multi_agent_report = self.multi_agent.evaluate(completed_steps)
        benchmark_report = self.benchmark.evaluate(performance_metrics, completed_steps, robustness_report)

        dimension_scores = {
            "planning": planning_report["score"],
            "tool_selection": tool_report["score"],
            "reasoning_trace": trace_report["score"],
            "robustness": _safe_float(robustness_report.get("score"), 0.0),
            "governance_safety": _safe_float(governance_report.get("score"), 0.0),
            "benchmark": benchmark_report["score"],
            "multi_agent": multi_agent_report["score"],
            "autonomy_control_balance": autonomy_report["score"],
        }
        weights = {
            "planning": 0.16,
            "tool_selection": 0.14,
            "reasoning_trace": 0.13,
            "robustness": 0.13,
            "governance_safety": 0.16,
            "benchmark": 0.12,
            "multi_agent": 0.08,
            "autonomy_control_balance": 0.08,
        }
        overall = round(sum(dimension_scores[k] * weights[k] for k in weights), 2)
        return {
            "framework_name": "Framework Evaluasi Agentic AI untuk Analisa dan Pengambilan Keputusan Digital Marketing",
            "goal": goal,
            "generated_at": datetime.now().isoformat(),
            "overall_agentic_score": overall,
            "dimension_scores": dimension_scores,
            "planning_evaluation": planning_report,
            "tool_selection_evaluation": tool_report,
            "reasoning_trace_evaluation": trace_report,
            "robustness_testing": robustness_report,
            "governance_safety_evaluation": governance_report,
            "benchmark_system": benchmark_report,
            "multi_agent_evaluation": multi_agent_report,
            "autonomy_control_balance": autonomy_report,
            "llm_stats": llm_stats,
            "interpretation": self._interpret(overall),
        }

    def _trace_evaluation(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        events = trajectory.get("events", []) if trajectory else []
        if not events:
            return {"score": 0.0, "event_count": 0, "interpretation": "No trajectory evidence was recorded."}
        fields = ["thought", "action", "observation", "reflection"]
        completeness = []
        for event in events:
            completeness.append(sum(bool(str(event.get(f, "")).strip()) for f in fields) / len(fields))
        success_ratio = sum(1 for e in events if e.get("success")) / max(1, len(events))
        score = round((statistics.mean(completeness) * 0.70 + success_ratio * 0.30) * 100, 2)
        return {
            "score": score,
            "event_count": len(events),
            "trajectory_path": trajectory.get("trajectory_path"),
            "trace_field_completeness_pct": round(statistics.mean(completeness) * 100, 2),
            "successful_trace_ratio_pct": round(success_ratio * 100, 2),
            "interpretation": "Reasoning/action/observation/reflection trace is publication-ready." if score >= 85 else "Trace exists but needs richer reflection/observation detail.",
        }

    def _autonomy_evaluation(self, llm_stats: Dict[str, Any], completed_steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        total_calls = _safe_float(llm_stats.get("llm_call_total"), 0.0)
        fallback_count = _safe_float(llm_stats.get("fallback_count"), 0.0)
        reliability = _safe_float(llm_stats.get("llm_reliability_pct"), 0.0)
        fallback_penalty = _pct(fallback_count, max(1.0, total_calls)) if total_calls else 0.0
        completion = _pct(sum(1 for s in completed_steps if s.get("success")), len(REQUIRED_WORKFLOW_TOOLS))
        score = round(reliability * 0.45 + completion * 0.40 + (100 - fallback_penalty) * 0.15, 2)
        return {
            "score": max(0.0, min(100.0, score)),
            "llm_reliability_pct": reliability,
            "fallback_penalty_pct": fallback_penalty,
            "completion_signal_pct": completion,
            "interpretation": "Agent autonomy is balanced by deterministic recovery." if score >= 75 else "Agent autonomy relies too heavily on fallback or incomplete execution.",
        }

    def _interpret(self, overall: float) -> str:
        if overall >= 85:
            return "Strong modern agentic AI evaluation framework; suitable for research demonstration."
        if overall >= 75:
            return "Good agentic AI framework; add broader benchmarks for stronger publication claims."
        if overall >= 60:
            return "Moderate agentic AI evidence; still partially resembles an ML pipeline."
        return "Weak agentic AI evidence; requires stronger trajectory, benchmark, and governance layers."


class AdaptiveWorkflowController:
    """State-aware controller used by the planner fallback and evaluation layer."""

    def suggest_next_action(self, state: Dict[str, Any], failed_tools: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        failed_tools = failed_tools or {}
        for tool in REQUIRED_WORKFLOW_TOOLS:
            if not state.get(tool):
                if failed_tools.get(tool, 0) >= 2:
                    return {
                        "tool": tool,
                        "reason": f"Adaptive retry: {tool} failed before; use safer defaults or HITL.",
                        "finish": False,
                        "strategy": "retry_with_safety_defaults",
                    }
                return {
                    "tool": tool,
                    "reason": f"Adaptive workflow: next unsatisfied capability is {tool}.",
                    "finish": False,
                    "strategy": "state_aware_next_step",
                }
        return {"tool": None, "reason": "All workflow capabilities are complete.", "finish": True, "strategy": "complete"}


class AgenticReportGenerator:
    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, evaluation: Dict[str, Any], trajectory: Dict[str, Any]) -> Dict[str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(self.output_dir, f"agentic_evaluation_report_{ts}.json")
        md_path = os.path.join(self.output_dir, f"agentic_evaluation_report_{ts}.md")
        chart_path = os.path.join(self.output_dir, f"agentic_evaluation_scores_{ts}.png")

        payload = {"evaluation": evaluation, "trajectory": trajectory}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._markdown(evaluation))

        chart_saved = self._save_chart(evaluation, chart_path)
        paths = {"json": json_path, "markdown": md_path}
        if chart_saved:
            paths["chart"] = chart_path
        return paths

    def _markdown(self, evaluation: Dict[str, Any]) -> str:
        lines = [
            f"# {evaluation.get('framework_name', 'Agentic AI Evaluation Report')}",
            "",
            f"Generated at: {evaluation.get('generated_at')}",
            "",
            f"Overall Agentic Score: **{evaluation.get('overall_agentic_score')} / 100**",
            "",
            f"Interpretation: {evaluation.get('interpretation')}",
            "",
            "## Dimension Scores",
            "",
            "| Dimension | Score |",
            "|---|---:|",
        ]
        for key, value in (evaluation.get("dimension_scores") or {}).items():
            lines.append(f"| {key.replace('_', ' ').title()} | {value} |")
        lines.extend([
            "",
            "## Planning Evaluation",
            "",
            f"{evaluation.get('planning_evaluation', {}).get('interpretation', '')}",
            "",
            "## Governance and Safety",
            "",
        ])
        gov = evaluation.get("governance_safety_evaluation", {})
        for control in gov.get("controls", []):
            lines.append(f"- Control: {control}")
        for risk in gov.get("risks", []):
            lines.append(f"- Risk: {risk}")
        lines.extend([
            "",
            "## Robustness Testing",
            "",
            f"{evaluation.get('robustness_testing', {}).get('interpretation', '')}",
        ])
        return "\n".join(lines) + "\n"

    def _save_chart(self, evaluation: Dict[str, Any], chart_path: str) -> bool:
        try:
            import matplotlib.pyplot as plt

            scores = evaluation.get("dimension_scores") or {}
            if not scores:
                return False
            labels = [k.replace("_", " ").title() for k in scores.keys()]
            values = [float(v) for v in scores.values()]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(labels, values)
            ax.set_ylim(0, 100)
            ax.set_ylabel("Score")
            ax.set_title("Agentic AI Evaluation Dimension Scores")
            ax.tick_params(axis="x", rotation=35)
            fig.tight_layout()
            fig.savefig(chart_path, dpi=150)
            plt.close(fig)
            return True
        except Exception:
            return False
