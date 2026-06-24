import logging
import os
import time
import json
import re
import ast

from dotenv import load_dotenv
from typing import Optional, Any, Dict, Tuple
from agents.data_loader_agent import DataLoaderAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.dynamic_analysis_agent import DynamicAnalysisAgent
from agents.optimization_agent import OptimizationAgent
from utils.hitl_interface import get_hitl_interface, HitlInterface
from utils.reporting import create_reporter
from utils.intelligent_summarization import create_summarizer
from utils.human_readable_report import print_human_readable_report
from utils.agentic_evaluation import (
    AgenticTrajectoryLogger,
    AgenticEvaluator,
    RobustnessTester,
    GovernanceSafetyEvaluator,
    AgenticReportGenerator,
    AdaptiveWorkflowController,
)
import pandas as pd
import numpy as np


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s'
)


class LLMPlannerAgent:
    """
    LLMPlannerAgent orchestrates the MAS workflow for
    digital marketing analytics and advertising decision support.
    """

    @staticmethod
    def _list_available_datasets():
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data'
        )

        datasets = []

        if not os.path.exists(data_dir):
            return datasets

        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)

            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.csv'):
                        datasets.append(
                            os.path.join(folder_path, file)
                        )

        return datasets

    @classmethod
    def interactive_setup(
        cls,
        hitl_interface: Optional[HitlInterface] = None
    ):
        if hitl_interface is None:
            hitl_interface = get_hitl_interface("cli")

        datasets = cls._list_available_datasets()

        dataset_path = hitl_interface.prompt_with_audit(
            "Select a dataset:",
            options=datasets,
            context={
                "step": "dataset_selection",
                "available_datasets": len(datasets)
            }
        )

        df = pd.read_csv(dataset_path, nrows=100)
        columns = list(df.columns)

        hitl_interface.show_info_with_audit(
            f"Columns in dataset ({len(columns)} total): {columns}",
            context={
                "step": "column_display",
                "dataset_path": dataset_path
            }
        )

        problem_types = [
            'classification',
            'regression'
        ]

        problem_type = hitl_interface.prompt_with_audit(
            "Select the problem type:",
            options=problem_types,
            context={
                "step": "problem_type_selection",
                "available_types": problem_types
            }
        )

        target_col = None

        if problem_type != 'anomaly_detection':
            target_col = hitl_interface.prompt_with_audit(
                "Select the target column (y):",
                options=columns,
                context={
                    "step": "target_selection",
                    "problem_type": problem_type
                }
            )

            columns = [
                col for col in columns
                if col != target_col
            ]

        feature_cols = hitl_interface.prompt_with_audit(
            "Select feature columns (Xs):",
            options=columns,
            multi_select=True,
            context={
                "step": "feature_selection",
                "problem_type": problem_type,
                "target_col": target_col
            }
        )

        return dataset_path, feature_cols, target_col, problem_type

    def __init__(
        self,
        dataset_path: str,
        feature_columns: list,
        target_column: str,
        problem_type: str,
        llm_agent: Optional[Any] = None,
        decision_llm_agent: Optional[Any] = None,
        hitl_interface: Optional[HitlInterface] = None
    ):
        logging.info("Initializing LLM Planner Agent...")

        load_dotenv()

        self.llm_agent = llm_agent
        self.decision_llm_agent = decision_llm_agent
        self.hitl_interface = hitl_interface or get_hitl_interface("cli")

        if self.llm_agent is None:
            # Default fallback: use Qwen via Ollama (local).
            from agents.local_llm_agent import LocalLLMAgent
            qwen_model = os.getenv("QWEN_MODEL_OVERRIDE", "qwen3:4b")
            self.llm_agent = LocalLLMAgent(backend="ollama", model_name=qwen_model)
            logging.info(
                f"No llm_agent provided. Falling back to Qwen local (Ollama): {qwen_model}"
            )

        else:
            logging.info("Using provided planner LLM agent for planning")

        self.dataset_path = dataset_path
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.problem_type = problem_type

        self.full_dataset = None
        self.raw_data = None
        self.preprocessed_data = None
        self.analysis_results = None
        self.recommendations = None

        self.failed_tools = {}

        # ── Comparison tracking ──────────────────────────────
        self._llm_call_total   = 0   # total LLM call attempts
        self._llm_call_success = 0   # calls that returned valid JSON
        self._llm_call_retry   = 0   # retry attempts (attempt > 0)
        self._fallback_count   = 0   # times deterministic fallback was used
        self._workflow_start_ts = None
        self._workflow_end_ts   = None

        self.reporter = create_reporter()
        self.summarizer = create_summarizer(
            self.decision_llm_agent
        )
        self.summarizer.set_logging_mode("minimal")

        self.tools = {
            "load_and_inspect_data": self._execute_perception_step,
            "preprocess_data": self._execute_preprocessing_step,
            "analyze_data": self._execute_analysis_step,
            "generate_recommendations": self._execute_optimization_step,
        }

        self._stage_snapshot: Dict[str, Dict[str, Any]] = {}

        # ── Agentic AI evaluation infrastructure ─────────────
        self.trajectory_logger = AgenticTrajectoryLogger(output_dir="logs")
        self.agentic_evaluator = AgenticEvaluator()
        self.robustness_tester = RobustnessTester()
        self.governance_evaluator = GovernanceSafetyEvaluator()
        self.agentic_report_generator = AgenticReportGenerator(output_dir="logs")
        self.adaptive_controller = AdaptiveWorkflowController()
        self.agentic_evaluation_results: Dict[str, Any] = {}
        self.agentic_report_paths: Dict[str, str] = {}
        self.robustness_report: Dict[str, Any] = {}
        self.governance_report: Dict[str, Any] = {}
        self._invalid_tool_decisions = 0
        self._latest_workflow_context: Dict[str, Any] = {}

    def run_workflow_with_llm(self, goal: str):
        logging.info(
            f"--- Starting LLM-driven Workflow with Goal: '{goal}' ---"
        )

        import time as _time
        self._workflow_start_ts = _time.time()

        self.reporter.log_step(
            "Workflow Initialization",
            True,
            f"Starting workflow with goal: {goal}"
        )

        self.reporter.report_data['workflow_goal'] = goal

        self.summarizer.store_workflow_start(
            dataset_path=self.dataset_path,
            problem_type=self.problem_type,
            target_column=self.target_column,
            feature_columns=self.feature_columns
        )

        available_tools = list(self.tools.keys())

        max_attempts = 3
        max_steps = len(available_tools) + 2

        workflow_context = {
            "goal": goal,
            "available_tools": available_tools,
            "completed_steps": [],
            "current_step": 0,
            "performance_insights": {},
            "strategy_adaptations": [],
            "learning_context": {
                "previous_approaches": [],
                "successful_patterns": [],
                "failed_patterns": []
            }
        }
        self._latest_workflow_context = workflow_context

        for step in range(max_steps):
            logging.info(
                f"--- LLM Turn {step + 1}/{max_steps} ---"
            )

            prompt = self._build_workflow_prompt(workflow_context)

            logging.info(
                f"Prompt sent to LLM:\n{prompt}\n---"
            )

            decision_source = "llm"
            decision = self._get_llm_decision(
                prompt,
                max_attempts
            )

            if decision is None:
                logging.warning(
                    "[LLM Planner] LLM failed. Using deterministic fallback."
                )
                self._fallback_count += 1
                decision_source = "deterministic_fallback"
                decision = self._deterministic_next_decision()

            if decision is None:
                logging.error(
                    "[LLM Planner] No valid decision available. Halting workflow."
                )
                break

            tool_name = decision.get('tool')
            finish_flag = decision.get('finish', False)
            reason = decision.get('reason', '')

            logging.info(
                f"[LLM Planner] Final decision: "
                f"tool='{tool_name}', finish={finish_flag}, reason='{reason}'"
            )

            if finish_flag and not tool_name:
                logging.info(
                    "[LLM Planner] Workflow is complete."
                )
                break

            if tool_name not in self.tools:
                logging.warning(
                    f"[LLM Planner] Invalid tool: '{tool_name}'. "
                    f"Available: {available_tools}"
                )

                self._invalid_tool_decisions += 1
                workflow_context["last_error"] = (
                    f"Invalid tool '{tool_name}' chosen. "
                    f"Must be one of: {available_tools}"
                )
                self.trajectory_logger.log_event(
                    step_number=step + 1,
                    action=str(tool_name),
                    thought=reason or decision.get("thought", "Invalid planner decision."),
                    observation=workflow_context["last_error"],
                    reflection="Planner selected an unavailable tool; validation rejected the action.",
                    success=False,
                    decision_source=decision_source,
                    metadata={"available_tools": available_tools, "raw_decision": decision}
                )

                continue

            logging.info(
                f"[LLM Planner] Executing tool: '{tool_name}'"
            )

            start_time = time.time()

            success, result_message = self.tools[tool_name]()

            duration = time.time() - start_time

            self.reporter.log_step(
                f"Tool Execution: {tool_name}",
                success,
                result_message,
                {
                    "tool": tool_name,
                    "reason": reason
                },
                duration
            )

            if success:
                self._emit_stage_summary(tool_name)

            logging.info(
                f"[LLM Planner] Tool '{tool_name}' result: "
                f"success={success}, message='{result_message}'"
            )

            thought_summary = decision.get("thought") or reason
            expected_observation = decision.get("expected_observation") or ""
            reflection = decision.get("reflection") or (
                f"Observed: {result_message}. "
                f"Next state will be updated based on success={success}."
            )

            step_data = {
                "tool": tool_name,
                "success": success,
                "message": result_message,
                "reason": reason,
                "thought": thought_summary,
                "expected_observation": expected_observation,
                "reflection": reflection,
                "decision_source": decision_source,
                "duration": duration,
                "step_number": step + 1
            }

            self.trajectory_logger.log_event(
                step_number=step + 1,
                action=tool_name,
                thought=thought_summary,
                observation=result_message,
                reflection=reflection,
                success=success,
                duration_seconds=duration,
                decision_source=decision_source,
                metadata={
                    "expected_observation": expected_observation,
                    "raw_decision": decision,
                }
            )

            workflow_context["completed_steps"].append(step_data)
            workflow_context["current_step"] = step + 1
            self._latest_workflow_context = workflow_context

            if success:
                workflow_context["learning_context"][
                    "successful_patterns"
                ].append({
                    "tool": tool_name,
                    "context": f"Step {step + 1}",
                    "outcome": "success"
                })

                workflow_context.pop("last_error", None)

            else:
                self.failed_tools[tool_name] = (
                    self.failed_tools.get(tool_name, 0) + 1
                )

                workflow_context["learning_context"][
                    "failed_patterns"
                ].append({
                    "tool": tool_name,
                    "context": f"Step {step + 1}",
                    "outcome": "failure",
                    "error": result_message
                })

                workflow_context["last_error"] = (
                    f"Tool '{tool_name}' failed: {result_message}"
                )

            if finish_flag:
                logging.info(
                    "[LLM Planner] LLM indicated workflow complete after tool."
                )
                break

        logging.info("--- LLM-driven Workflow Finished ---")

        self._run_agentic_evaluation(goal, workflow_context)

        self.summarizer.store_workflow_end()

        # Simpan ringkasan internal ke file JSON (tidak dicetak ulang)
        detailed_results_path = (
            self.summarizer.save_detailed_results()
        )
        logging.info(
            f"📄 Detailed results saved to: {detailed_results_path}"
        )

        import time as _time
        self._workflow_end_ts = _time.time()

        # Generate summary before saving so comparison metrics are populated.
        try:
            self.reporter.generate_summary()
        except Exception as _summary_err:
            logging.warning(f"[Reporter] Summary generation failed: {_summary_err}")

        report_path = self.reporter.save_report()
        publication_paths = self.reporter.save_publication_snapshot()

        logging.info(
            "LLM-powered MAS application has finished its run."
        )

        logging.info(
            f"📁 Publication snapshot: {publication_paths['json']}"
        )

        if publication_paths.get('csv'):
            logging.info(
                f"📁 Publication recommendations CSV: "
                f"{publication_paths['csv']}"
            )

        # =====================================================
        # LAPORAN MUDAH DIBACA (Human-Readable Report)
        # =====================================================
        try:
            perf_metrics = self.reporter.report_data.get(
                'performance_metrics', {}
            )

            recs_df = getattr(self, 'recommendations', None)

            feat_imp = None
            if self.analysis_results and 'feature_importances' in self.analysis_results:
                feat_imp = self.analysis_results['feature_importances']

            workflow_start = self.reporter.report_data.get('workflow_start')
            workflow_end   = self.reporter.report_data.get('workflow_end')
            duration_sec   = None
            if workflow_start and workflow_end:
                from datetime import datetime as _dt
                try:
                    t0 = _dt.fromisoformat(workflow_start)
                    t1 = _dt.fromisoformat(workflow_end)
                    duration_sec = (t1 - t0).total_seconds()
                except Exception:
                    pass

            print_human_readable_report(
                performance_metrics=perf_metrics,
                recommendations_df=recs_df,
                feature_importances=feat_imp,
                duration_seconds=duration_sec,
            )
        except Exception as _hr_err:
            logging.warning(
                f"[Human-Readable Report] Gagal menghasilkan laporan: {_hr_err}"
            )

    def _run_agentic_evaluation(
        self,
        goal: str,
        workflow_context: Dict[str, Any]
    ):
        """
        Run the dedicated Agentic AI evaluation layer.

        This is intentionally separated from model performance so the project
        evaluates the agent as an autonomous, auditable workflow rather than
        only as a prediction pipeline.
        """
        try:
            completed_steps = workflow_context.get("completed_steps", [])
            performance_metrics = self.reporter.report_data.get(
                'performance_metrics', {}
            )

            self.robustness_report = self.robustness_tester.run(
                raw_data=self.raw_data,
                preprocessed_data=self.preprocessed_data,
                analysis_results=self.analysis_results,
                task=self.problem_type,
            )

            dataset_columns = []
            try:
                if self.full_dataset is not None:
                    dataset_columns = list(self.full_dataset.columns)
                elif self.raw_data is not None:
                    dataset_columns = list(self.raw_data.columns)
            except Exception:
                dataset_columns = []

            self.governance_report = self.governance_evaluator.evaluate(
                recommendations=self.recommendations,
                hitl_events=self.reporter.report_data.get('hitl_events', []),
                audit_log_path=getattr(self.hitl_interface, 'audit_log_path', None),
                trajectory_path=self.trajectory_logger.path,
                dataset_columns=dataset_columns,
            )

            llm_stats = {
                "llm_call_total": self._llm_call_total,
                "llm_call_success": self._llm_call_success,
                "llm_call_retry": self._llm_call_retry,
                "fallback_count": self._fallback_count,
                "llm_reliability_pct": self._decision_reliability_score(),
            }

            trajectory = self.trajectory_logger.to_dict()
            self.agentic_evaluation_results = self.agentic_evaluator.evaluate(
                goal=goal,
                completed_steps=completed_steps,
                trajectory=trajectory,
                llm_stats=llm_stats,
                performance_metrics=performance_metrics,
                robustness_report=self.robustness_report,
                governance_report=self.governance_report,
                invalid_decisions=self._invalid_tool_decisions,
            )

            self.agentic_report_paths = self.agentic_report_generator.save(
                self.agentic_evaluation_results,
                trajectory,
            )

            self.reporter.report_data['agentic_evaluation'] = (
                self.agentic_evaluation_results
            )
            self.reporter.report_data['agentic_report_paths'] = (
                self.agentic_report_paths
            )
            self.summarizer.stored_results['agentic_evaluation'] = (
                self.agentic_evaluation_results
            )
            self.summarizer.stored_results['agentic_report_paths'] = (
                self.agentic_report_paths
            )

            logging.info(
                "🧭 Agentic evaluation complete: "
                f"overall_score={self.agentic_evaluation_results.get('overall_agentic_score')}"
            )
            logging.info(
                f"📄 Agentic evaluation report: {self.agentic_report_paths.get('json')}"
            )
        except Exception as exc:
            logging.warning(
                f"[Agentic Evaluation] Failed to run evaluation layer: {exc}",
                exc_info=True
            )
            self.agentic_evaluation_results = {
                "overall_agentic_score": 0,
                "error": str(exc),
            }

    def get_comparison_metrics(self) -> dict:
        """Return all metrics needed for side-by-side comparison report."""
        import time as _time

        duration = None
        if self._workflow_start_ts and self._workflow_end_ts:
            duration = round(self._workflow_end_ts - self._workflow_start_ts, 2)

        perf = self.reporter.report_data.get('performance_metrics', {})
        summary = self.reporter.report_data.get('summary', {})
        recs = getattr(self, 'recommendations', None)

        rec_dist = {}
        rec_channel = {}
        rec_priority = {}
        total_recs = 0
        if recs is not None and hasattr(recs, 'empty') and not recs.empty:
            total_recs = len(recs)
            if 'Decision_Area' in recs.columns:
                rec_dist = recs['Decision_Area'].value_counts().to_dict()
            if 'CampaignChannel' in recs.columns:
                rec_channel = recs['CampaignChannel'].value_counts().to_dict()
            if 'Priority_Level' in recs.columns:
                rec_priority = recs['Priority_Level'].value_counts().to_dict()

        steps_summary = summary.get('total_steps', 0)
        success_steps = summary.get('successful_steps', 0)

        reliability_score = self._decision_reliability_score()

        agentic = self.agentic_evaluation_results or {}
        dim = agentic.get("dimension_scores", {}) if isinstance(agentic, dict) else {}

        def _metric_pct(name):
            value = perf.get(name, 0)
            try:
                if value is None:
                    return 0
                return round(float(value) * 100, 2)
            except Exception:
                return 0

        return {
            # ── Model performance ──────────────────────
            'accuracy':  _metric_pct('accuracy'),
            'precision': _metric_pct('precision'),
            'recall':    _metric_pct('recall'),
            'f1_score':  _metric_pct('f1_score'),
            'roc_auc':   _metric_pct('roc_auc'),
            'model_used': perf.get('model_used', 'N/A'),
            # ── Workflow execution ─────────────────────
            'duration_seconds': duration,
            'total_steps':   steps_summary,
            'success_steps': success_steps,
            'task_completion_rate': round(
                (success_steps / steps_summary * 100) if steps_summary else 0, 1
            ),
            # ── LLM orchestrator reliability ───────────
            'llm_call_total':   self._llm_call_total,
            'llm_call_success': self._llm_call_success,
            'llm_call_retry':   self._llm_call_retry,
            'fallback_count':   self._fallback_count,
            'llm_reliability_pct': reliability_score,
            'fallback_rate_pct': round(
                (self._fallback_count / max(self._llm_call_total, 1)) * 100, 2
            ),
            # ── Recommendations ────────────────────────
            'total_recommendations': total_recs,
            'rec_by_area':     rec_dist,
            'rec_by_channel':  rec_channel,
            'rec_by_priority': rec_priority,
            # ── Agentic AI evaluation ──────────────────
            'agentic_overall_score': round(agentic.get('overall_agentic_score', 0), 2) if isinstance(agentic, dict) else 0,
            'planning_score': round(dim.get('planning', 0), 2),
            'tool_selection_score': round(dim.get('tool_selection', 0), 2),
            'reasoning_trace_score': round(dim.get('reasoning_trace', 0), 2),
            'robustness_score': round(dim.get('robustness', 0), 2),
            'governance_safety_score': round(dim.get('governance_safety', 0), 2),
            'benchmark_score': round(dim.get('benchmark', 0), 2),
            'multi_agent_score': round(dim.get('multi_agent', 0), 2),
            'autonomy_control_score': round(dim.get('autonomy_control_balance', 0), 2),
            'trajectory_log_path': self.trajectory_logger.path,
            'agentic_report_paths': self.agentic_report_paths,
        }

    def _deterministic_next_decision(self):
        """
        Deterministic fallback planner.

        This makes the workflow stable even when the local LLM returns
        verbose or invalid JSON.
        """

        state = {
            "load_and_inspect_data": self.raw_data is not None,
            "preprocess_data": self.preprocessed_data is not None,
            "analyze_data": self.analysis_results is not None,
            "generate_recommendations": self.recommendations is not None,
        }
        decision = self.adaptive_controller.suggest_next_action(
            state,
            failed_tools=self.failed_tools
        )
        decision.setdefault("thought", decision.get("reason", "Adaptive fallback selected the next action."))
        decision.setdefault("expected_observation", "The selected tool should satisfy the next incomplete workflow capability.")
        decision.setdefault("reflection", "Fallback uses explicit workflow state to preserve task progress.")
        return decision

    def _decision_reliability_score(self) -> float:
        """
        Reliability is reported at workflow-decision level.

        Retry attempts are still tracked separately in llm_call_retry, but they
        should not dominate the headline score because one logical planner turn
        can contain multiple repair attempts.
        """

        total_decisions = self._llm_call_success + self._fallback_count

        if total_decisions <= 0:
            return 0.0

        return round(
            self._llm_call_success / total_decisions * 100,
            2
        )

    def _build_workflow_prompt(
        self,
        context: Dict[str, Any]
    ) -> str:
        prompt_parts = [
            f"Goal: {context['goal']}",
            f"Available tools: {context['available_tools']}",
            f"Current step: {context['current_step'] + 1}",
            "",
            "DOMAIN CONTEXT:",
            "- This workflow is for digital marketing and advertising analytics.",
            "- The main objective is campaign performance analysis, conversion prediction, and decision recommendation.",
            "- Recommendations should focus on campaign optimization, channel prioritization, audience targeting, budget allocation, and engagement improvement.",
            ""
        ]

        if context.get('completed_steps'):
            prompt_parts.append("Completed steps:")

            for i, step in enumerate(context['completed_steps']):
                status = "SUCCESS" if step['success'] else "FAILED"

                duration_str = (
                    f" ({step.get('duration', 0):.2f}s)"
                    if 'duration' in step
                    else ""
                )

                prompt_parts.append(
                    f"  {i + 1}. {status} "
                    f"{step['tool']}: "
                    f"{step['message']}{duration_str}"
                )

        if context.get('last_error'):
            prompt_parts.append(
                f"\nLast error: {context['last_error']}"
            )

        prompt_parts.extend([
            "",
            "TASK:",
            "Choose the next tool required to complete the workflow.",
            "You must respond with a valid JSON object.",
            "",
            "REASONING GUIDANCE:",
            "1. If no data has been loaded, choose load_and_inspect_data.",
            "2. If data is loaded but not preprocessed, choose preprocess_data.",
            "3. If data is preprocessed but not analyzed, choose analyze_data.",
            "4. If analysis is complete but recommendations are not generated, choose generate_recommendations.",
            "5. If recommendations are generated, set finish=true.",
            "",
            "OUTPUT FORMAT:",
            '{"tool":"<tool_name>","thought":"<brief visible rationale>","reason":"<short reason>","expected_observation":"<what should change>","reflection":"<how this step supports the goal>","finish":false}',
            "",
            "VALID TOOLS:",
            "- load_and_inspect_data",
            "- preprocess_data",
            "- analyze_data",
            "- generate_recommendations",
            "",
            "Rules:",
            "CRITICAL JSON RULES:",
            "- Return ONLY valid JSON.",
            "- No markdown.",
            "- No explanation outside JSON.",
            "- Do not reveal hidden chain-of-thought; use only a short, safe thought summary inside the JSON field named thought.",
            "- No analysis outside JSON.",
            "- No ```json blocks.",
            "- Response must start with {.",
            "- Response must end with }.",
            "",
            "VALID RESPONSE EXAMPLES:",
            '{"tool":"load_and_inspect_data","thought":"Need perception before any decision.","reason":"The dataset must be loaded before preprocessing.","expected_observation":"Dataset shape and columns are known.","reflection":"This creates an auditable perception step.","finish":false}',
            '{"tool":"preprocess_data","thought":"Raw campaign data needs model-ready features.","reason":"The loaded dataset must be prepared before model analysis.","expected_observation":"Clean feature matrix is available.","reflection":"This reduces data quality risk before analysis.","finish":false}',
            '{"tool":"analyze_data","thought":"Prepared data should be benchmarked.","reason":"The preprocessed data is ready for model analysis.","expected_observation":"Model benchmark and performance metrics are produced.","reflection":"This supports benchmark-based agentic evaluation.","finish":false}',
            '{"tool":"generate_recommendations","thought":"Model findings must become marketing decisions.","reason":"Model analysis is complete and recommendations are needed.","expected_observation":"Recommendations are generated and reviewed by HITL.","reflection":"This links analysis to governed decision support.","finish":false}',
            '{"tool":null,"thought":"All required capabilities are complete.","reason":"The workflow has completed successfully.","expected_observation":"No further tool is needed.","reflection":"The trajectory can now be evaluated.","finish":true}'
        ])

        return "\n".join(prompt_parts)

    def _get_llm_decision(
        self,
        prompt: str,
        max_attempts: int = 3
    ):
        """
        Get structured decision from LLM.

        Robust for verbose Ollama/Qwen outputs.
        """

        retry_prompt = prompt

        for attempt in range(max_attempts):
            if attempt > 0:
                self._llm_call_retry += 1
            self._llm_call_total += 1
            try:
                if self.llm_agent is not None:
                    response = self.llm_agent.generate(
                        retry_prompt,
                        max_tokens=512
                    )
                    raw_text = response.get('raw', '')

                    logging.info(
                        f"Local LLM response "
                        f"(attempt {attempt + 1}): {raw_text}"
                    )

                    decision = self._parse_json_response(raw_text)

                    if decision is None and isinstance(response, dict):
                        parsed_from_adapter = response.get('parsed')

                        if isinstance(parsed_from_adapter, dict):
                            decision = parsed_from_adapter

                    logging.info(
                        f"RAW DECISION RESULT: {decision}"
                    )

                else:
                    # llm_agent is always set; this branch is a safety fallback.
                    logging.warning("llm_agent unexpectedly None; returning empty decision.")
                    decision = {}

                logging.info(
                    f"PARSED DECISION attempt {attempt + 1}: {decision}"
                )

                decision = self._normalize_decision(decision)

                if self._validate_decision(decision):
                    self._llm_call_success += 1
                    return decision

                logging.warning(
                    f"Invalid decision on attempt "
                    f"{attempt + 1}: {decision}"
                )

                retry_prompt = self._build_repair_prompt(
                    prompt,
                    raw_text,
                    decision
                )

            except Exception as e:
                logging.warning(
                    f"Error getting LLM decision on attempt "
                    f"{attempt + 1}: {e}"
                )

        return None

    def _build_repair_prompt(
        self,
        original_prompt: str,
        raw_text: str,
        decision: Optional[Dict[str, Any]]
    ) -> str:
        expected = self._expected_next_tool()

        repair_parts = [
            original_prompt,
            "",
            "PREVIOUS RESPONSE WAS INVALID.",
            "Return exactly one JSON object now.",
            f"Expected next tool based on workflow state: {expected}",
            f"Invalid parsed decision: {decision}",
            f"Previous raw response excerpt: {str(raw_text)[:500]}",
            "",
            "Return only this shape:",
            '{"tool":"<valid_tool_or_null>","thought":"short visible rationale","reason":"short reason","expected_observation":"expected state change","reflection":"why this is useful","finish":false}'
        ]

        return "\n".join(repair_parts)

    def _expected_next_tool(self) -> Optional[str]:
        if self.raw_data is None:
            return "load_and_inspect_data"
        if self.preprocessed_data is None:
            return "preprocess_data"
        if self.analysis_results is None:
            return "analyze_data"
        if self.recommendations is None:
            return "generate_recommendations"
        return None

    def _validate_decision(self, decision: Dict[str, Any]) -> bool:
        """Validate a planner decision object used by tests and runtime guards."""
        if not isinstance(decision, dict):
            return False
        if "finish" not in decision:
            return False
        finish = decision.get("finish")
        if isinstance(finish, str):
            finish = finish.strip().lower() == "true"
        if finish is True:
            return decision.get("tool") in (None, "", *self._available_tool_names())
        tool = decision.get("tool")
        return isinstance(tool, str) and tool in self._available_tool_names()

    def _normalize_decision(
        self,
        decision: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(decision, dict):
            return None

        normalized = dict(decision)

        if "tool_name" in normalized and "tool" not in normalized:
            normalized["tool"] = normalized["tool_name"]

        if "action" in normalized and "tool" not in normalized:
            normalized["tool"] = normalized["action"]

        if "finish" not in normalized:
            done = normalized.get("done", normalized.get("complete", False))
            normalized["finish"] = done

        if isinstance(normalized.get("finish"), str):
            normalized["finish"] = (
                normalized["finish"].strip().lower()
                in ("true", "yes", "1", "complete", "completed", "done")
            )

        if normalized.get("finish") is True:
            tool = normalized.get("tool")
            if tool in ("", "none", "null", "None", "NULL"):
                normalized["tool"] = None
            normalized.setdefault("reason", "Workflow complete.")
            normalized.setdefault("thought", normalized["reason"])
            normalized.setdefault("expected_observation", "No further tool is needed.")
            normalized.setdefault("reflection", "The workflow can be finalized.")
            return normalized

        tool = normalized.get("tool")
        if tool is not None:
            normalized["tool"] = self._canonical_tool_name(str(tool))

        normalized.setdefault("reason", "No reason provided.")
        normalized.setdefault("thought", normalized.get("reason", "Planner selected the next tool."))
        normalized.setdefault("expected_observation", "The workflow advances to the next required state.")
        normalized.setdefault("reflection", "The selected action supports the workflow goal.")

        return normalized

    def _canonical_tool_name(self, tool_name: str) -> str:
        cleaned = (
            tool_name.strip()
            .replace("`", "")
            .replace('"', "")
            .replace("'", "")
            .lower()
        )

        exact = {
            name.lower(): name
            for name in self._available_tool_names()
        }

        if cleaned in exact:
            return exact[cleaned]

        aliases = {
            "load_data": "load_and_inspect_data",
            "inspect_data": "load_and_inspect_data",
            "load_dataset": "load_and_inspect_data",
            "data_loading": "load_and_inspect_data",
            "perception": "load_and_inspect_data",
            "preprocess": "preprocess_data",
            "preprocessing": "preprocess_data",
            "clean_data": "preprocess_data",
            "feature_engineering": "preprocess_data",
            "analyze": "analyze_data",
            "analysis": "analyze_data",
            "model_data": "analyze_data",
            "train_model": "analyze_data",
            "model_analysis": "analyze_data",
            "generate_recommendation": "generate_recommendations",
            "recommend": "generate_recommendations",
            "recommendations": "generate_recommendations",
            "optimization": "generate_recommendations",
            "optimize": "generate_recommendations",
        }

        if cleaned in aliases:
            return aliases[cleaned]

        compact = re.sub(r"[^a-z_]+", "_", cleaned).strip("_")

        if compact in exact:
            return exact[compact]

        if compact in aliases:
            return aliases[compact]

        return tool_name

    def _parse_json_response(
        self,
        text: str
    ):
        """
        Robust JSON parser for Qwen/Ollama outputs.

        Handles:
        - <think> blocks
        - markdown fences
        - nested JSON
        - verbose explanations
        - multiple JSON objects
        """

        if not text:
            return None

        try:
            cleaned = str(text)

            # Remove markdown fences
            cleaned = cleaned.replace("```json", "")
            cleaned = cleaned.replace("```", "")
            cleaned = cleaned.replace("`", "")

            # Remove common reasoning markers
            cleaned = cleaned.replace("Thinking...", "")
            cleaned = cleaned.replace("...done thinking.", "")

            # Remove Qwen think blocks
            cleaned = re.sub(
                r"<think>.*?</think>",
                "",
                cleaned,
                flags=re.DOTALL | re.IGNORECASE
            )

            cleaned = cleaned.strip()

            # If a model starts a think block but does not close it, keep the
            # part from the first JSON-like object onward.
            if "<think>" in cleaned.lower() and "</think>" not in cleaned.lower():
                first_json = cleaned.find("{")

                if first_json >= 0:
                    cleaned = cleaned[first_json:]

            logging.info(
                f"JSON PARSER INPUT: {cleaned[:500]}"
            )

            # Find ALL balanced JSON objects
            candidates = []

            start_positions = [
                i for i, ch in enumerate(cleaned)
                if ch == "{"
            ]

            for start in start_positions:

                depth = 0

                for end in range(start, len(cleaned)):

                    if cleaned[end] == "{":
                        depth += 1

                    elif cleaned[end] == "}":
                        depth -= 1

                        if depth == 0:
                            candidate = cleaned[start:end + 1]
                            candidates.append(candidate)
                            break

            if not candidates:

                logging.warning(
                    "No JSON object found in LLM response."
                )

                return self._extract_decision_from_text(cleaned)

            # Try from last candidate first
            for candidate in reversed(candidates):

                try:

                    parsed = json.loads(candidate)

                    if isinstance(parsed, dict):

                        logging.info(
                            f"JSON PARSED SUCCESSFULLY: {parsed}"
                        )

                        return parsed

                except Exception:
                    parsed = self._parse_relaxed_json(candidate)

                    if isinstance(parsed, dict):

                        logging.info(
                            f"RELAXED JSON PARSED SUCCESSFULLY: {parsed}"
                        )

                        return parsed

                    continue

            logging.warning(
                "JSON objects found but none were parseable."
            )

            return self._extract_decision_from_text(cleaned)

        except Exception as e:

            logging.warning(
                f"JSON parsing failed: {e}"
            )

            return self._extract_decision_from_text(str(text))

    def _parse_relaxed_json(
        self,
        candidate: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse common non-strict JSON variants emitted by local LLMs.
        """

        if not candidate:
            return None

        variants = []
        stripped = candidate.strip()
        variants.append(stripped)

        # Python-literal style: {'tool': '...', 'finish': False}
        variants.append(
            stripped
            .replace(": true", ": True")
            .replace(": false", ": False")
            .replace(": null", ": None")
        )

        # Trailing commas before object close are common in generated JSON.
        variants.append(
            re.sub(r",\s*}", "}", stripped)
        )

        for variant in variants:
            try:
                parsed = ast.literal_eval(variant)

                if isinstance(parsed, dict):
                    return parsed

            except Exception:
                continue

        return None

    def _extract_decision_from_text(
        self,
        text: str
    ) -> Optional[Dict[str, Any]]:
        """
        Last-resort extraction for outputs that explain the decision in text.

        This still uses the LLM response as the source; it only maps an
        explicitly mentioned tool/action into the structured planner contract.
        """

        if not text:
            return None

        lower = text.lower()

        finish = any(
            phrase in lower
            for phrase in [
                "finish true",
                "finish: true",
                '"finish": true',
                "workflow complete",
                "no further tool",
                "all required capabilities are complete",
            ]
        )

        mentioned_tools = []

        for tool in self._available_tool_names():
            if tool.lower() in lower:
                mentioned_tools.append(tool)

        if not mentioned_tools:
            alias_patterns = [
                ("load_and_inspect_data", r"\b(load|inspect|perception)\b"),
                ("preprocess_data", r"\b(preprocess|clean|feature engineering|transform)\b"),
                ("analyze_data", r"\b(analyze|analysis|train|model|benchmark)\b"),
                ("generate_recommendations", r"\b(recommend|recommendation|optimi[sz]e|decision)\b"),
            ]

            for tool, pattern in alias_patterns:
                if re.search(pattern, lower):
                    mentioned_tools.append(tool)
                    break

        if finish and not mentioned_tools:
            return {
                "tool": None,
                "thought": "All required workflow capabilities appear complete.",
                "reason": "The LLM indicated that the workflow is complete.",
                "expected_observation": "No further tool is needed.",
                "reflection": "The workflow can be finalized.",
                "finish": True,
            }

        if not mentioned_tools:
            return None

        expected = self._expected_next_tool()
        tool = expected if expected in mentioned_tools else mentioned_tools[-1]

        return {
            "tool": tool,
            "thought": f"The LLM response indicated {tool} as the next action.",
            "reason": f"Recovered tool decision from non-JSON LLM output: {tool}.",
            "expected_observation": "The selected tool should advance the workflow state.",
            "reflection": "Recovered structured action from the model response.",
            "finish": False,
        }

    def _available_tool_names(self) -> list:
        if isinstance(self.tools, dict):
            return list(self.tools.keys())

        if isinstance(self.tools, (list, tuple, set)):
            return list(self.tools)

        return []

    def _emit_stage_summary(
        self,
        tool_name: str
    ):
        snapshot = self._stage_snapshot.pop(
            tool_name,
            None
        )

        if snapshot:
            self.reporter.log_stage_summary(
                snapshot.get("title", tool_name),
                snapshot.get("summary", ""),
                snapshot.get("stats", {})
            )

    def _execute_perception_step(self):
        agent = DataLoaderAgent(self.dataset_path)

        self.raw_data = agent.load_data()

        if self.raw_data is None:
            return False, "Data loading failed."

        self.full_dataset = self.raw_data.copy()

        if self.problem_type == 'anomaly_detection':
            self.raw_data = self.raw_data[self.feature_columns]

        else:
            self.raw_data = self.raw_data[
                self.feature_columns + [self.target_column]
            ]

        agent.inspect_data()

        feature_list = ", ".join(self.feature_columns[:5])

        if len(self.feature_columns) > 5:
            feature_list += ", ..."

        stats = {
            "Rows": f"{self.raw_data.shape[0]:,}",
            "Columns": self.raw_data.shape[1],
            "Selected Features": feature_list
        }

        if self.problem_type != 'anomaly_detection':
            stats["Target"] = self.target_column

        dataset_name = os.path.basename(self.dataset_path)

        summary = (
            f"Loaded '{dataset_name}' for {self.problem_type} "
            "with selected digital marketing features."
        )

        self._stage_snapshot["load_and_inspect_data"] = {
            "title": "Stage · Data Loading & Inspection",
            "summary": summary,
            "stats": stats
        }

        return True, f"Data loaded. Shape: {self.raw_data.shape}"

    def _execute_preprocessing_step(self):
        if self.raw_data is None:
            return False, "Cannot preprocess, raw_data is not loaded."

        if self.problem_type == 'anomaly_detection':
            analysis_data = self.raw_data
            target_col = None
            protected_cols = list(self.raw_data.columns)

        else:
            analysis_data = self.raw_data[
                self.feature_columns + [self.target_column]
            ]

            target_col = self.target_column
            protected_cols = self.feature_columns

        agent = PreprocessingAgent(
            analysis_data,
            target_column=target_col,
            problem_type=self.problem_type,
            protected_columns=protected_cols
        )

        processed_features = agent.preprocess()

        if processed_features is None:
            return False, "Preprocessing failed."

        if (
            hasattr(agent, 'last_feature_insights')
            and agent.last_feature_insights
        ):
            self.reporter.log_feature_analysis(
                agent.last_feature_insights
            )

            self.summarizer.store_feature_analysis(
                agent.last_feature_insights
            )

        if (
            self.problem_type != 'anomaly_detection'
            and self.target_column is not None
        ):
            target = self.raw_data[[self.target_column]]

            self.preprocessed_data = pd.concat(
                [
                    processed_features,
                    target
                ],
                axis=1
            )

        else:
            self.preprocessed_data = processed_features

        processed_rows, processed_cols = self.preprocessed_data.shape

        numeric_cols = self.preprocessed_data.select_dtypes(
            include=[np.number]
        ).shape[1]

        categorical_cols = processed_cols - numeric_cols

        summary = (
            f"Preprocessed data ready "
            f"({processed_rows:,} rows x {processed_cols} columns) "
            f"for {self.problem_type} stage."
        )

        stats = {
            "Numeric Columns": numeric_cols,
            "Non-numeric Columns": categorical_cols
        }

        if (
            self.problem_type != 'anomaly_detection'
            and self.target_column
        ):
            stats["Target Appended"] = "Yes"

        self._stage_snapshot["preprocess_data"] = {
            "title": "Stage · Marketing Feature Engineering",
            "summary": summary,
            "stats": stats
        }

        return True, (
            f"Preprocessing complete. "
            f"Shape: {self.preprocessed_data.shape}"
        )

    def _execute_analysis_step(self):
        if self.preprocessed_data is None:
            return False, "Cannot analyze, data is not preprocessed."

        # ── Progress indicator sebelum analisis dimulai ──────
        print()
        print("━" * 62)
        print("  ⏳  Memulai analisis model — mohon tunggu...")
        print("━" * 62)
        print()

        params = None

        agent = DynamicAnalysisAgent(
            self.preprocessed_data,
            self.target_column,
            task=self.problem_type,
            params=params
        )

        # Selalu jalankan benchmark penuh (semua model)
        results = agent.run_full_benchmark()

        adaptive_intelligence_used = False
        tried_models = list(
            r["model"] for r in (results.get("benchmark_table") or [])
            if "model" in r
        ) if results else []

        if results and self._is_poor_performance(results):
            metric_label, metric_value = self._performance_metric(results)

            if metric_label and metric_value is not None:
                self.hitl_interface.show_warning_with_audit(
                    f"Model performance is low "
                    f"({metric_label} = {metric_value:.3f}).",
                    context={
                        "step": "low_performance_warning",
                        "metric_label": metric_label,
                        "metric_value": float(metric_value),
                        "task": self.problem_type
                    }
                )

                decision = self.hitl_interface.prompt_with_audit(
                    "Would you like to retry with alternative models?",
                    options=["retry", "proceed"],
                    context={
                        "step": "low_performance_decision",
                        "metric_label": metric_label,
                        "metric_value": float(metric_value)
                    }
                ).lower()

                if decision == "retry":
                    tried_models = (
                        agent.tried_models.copy()
                        if hasattr(agent, 'tried_models')
                        else []
                    )

                    results = agent.run(force_retry=True)
                    adaptive_intelligence_used = True

                    if results is None:
                        return False, (
                            "Dynamic analysis failed after retry."
                        )

        if results is None:
            return False, "Dynamic analysis failed."

        performance_metrics = {}

        for metric in [
            'accuracy',
            'precision',
            'recall',
            'f1_score',
            'roc_auc',
            'r2',
            'mse',
            'rmse'
        ]:
            if metric in results:
                performance_metrics[metric] = results[metric]

        if 'model' in results:
            performance_metrics['model_used'] = results['model']

        # Simpan benchmark_table ke performance_metrics untuk laporan
        if results.get('benchmark_table'):
            performance_metrics['benchmark_table'] = results['benchmark_table']

        if performance_metrics:
            self.reporter.log_performance_metrics(
                performance_metrics
            )

        self.summarizer.store_model_result(
            model_name=results.get('model', 'Unknown'),
            performance=performance_metrics,
            adaptive_intelligence=adaptive_intelligence_used,
            tried_models=tried_models
        )

        feature_importances = None

        if (
            results.get('feature_importances') is not None
            and results.get('feature_names') is not None
        ):
            feature_importances = pd.DataFrame({
                'feature': results['feature_names'],
                'importance': results['feature_importances']
            }).sort_values(
                by='importance',
                ascending=False
            )

        if self.problem_type == 'anomaly_detection':
            self.analysis_results = {
                'evaluation': results
            }

        else:
            self.analysis_results = {
                'evaluation': results,
                'feature_importances': feature_importances,
                'test_data_features': results.get('X_test'),
                'test_predictions': results.get('predictions'),
                'train_predictions': results.get('train_predictions')
            }

        if self.problem_type == 'regression':
            msg = (
                f"Dynamic analysis complete. "
                f"Model: {results.get('model')}, "
                f"R²: {results.get('r2')}, "
                f"MSE: {results.get('mse')}"
            )

        elif self.problem_type == 'classification':
            msg = (
                f"Dynamic analysis complete. "
                f"Model: {results.get('model')}, "
                f"Accuracy: {results.get('accuracy')}"
            )

        else:
            msg = (
                f"Dynamic analysis complete. "
                f"Model: {results.get('model')}"
            )

        model_name = results.get('model', 'Unknown')

        analysis_stats = {
            "Model": model_name,
            "Samples Evaluated": f"{self.preprocessed_data.shape[0]:,}",
            "Adaptive Retry": (
                "Yes" if adaptive_intelligence_used else "No"
            )
        }

        if self.problem_type == 'classification':
            accuracy_val = results.get('accuracy')

            if isinstance(accuracy_val, (int, float)):
                analysis_stats["Accuracy"] = f"{accuracy_val:.4f}"

        elif self.problem_type == 'regression':
            r2_val = results.get('r2')
            mse_val = results.get('mse')

            if isinstance(r2_val, (int, float)):
                analysis_stats["R²"] = f"{r2_val:.4f}"

            if isinstance(mse_val, (int, float)):
                analysis_stats["MSE"] = f"{mse_val:.6f}"

        self._stage_snapshot["analyze_data"] = {
            "title": "Stage · Marketing Modeling & Evaluation",
            "summary": (
                f"Completed model analysis for {self.problem_type} "
                "digital marketing task."
            ),
            "stats": analysis_stats
        }

        return True, msg

    def _execute_optimization_step(self):
        if self.analysis_results is None:
            return False, "Cannot optimize, analysis not done."

        if self.problem_type == 'anomaly_detection':
            payload = {
                'results_df': self.analysis_results[
                    'evaluation'
                ].get('results_df'),
                'anomaly_labels': self.analysis_results[
                    'evaluation'
                ].get('anomaly_labels')
            }

        else:
            context = self.raw_data.loc[
                self.analysis_results[
                    'test_data_features'
                ].index
            ]

            payload = {
                'test_data': context,
                'test_predictions': self.analysis_results[
                    'test_predictions'
                ],
                'train_predictions': self.analysis_results.get(
                    'train_predictions'
                ),
                'feature_importances': self.analysis_results[
                    'feature_importances'
                ]
            }

        agent = OptimizationAgent(payload)

        recommendations = agent.generate_recommendations()

        if recommendations is None:
            return False, "Optimization failed."

        summary_report = agent.generate_summary_report(
            recommendations
        )

        self.reporter.log_recommendations({
            'recommendations': recommendations,
            'summary_report': summary_report
        })

        self.summarizer.store_recommendations({
            'recommendations': recommendations,
            'summary_report': summary_report
        })

        # ── HUMAN-IN-THE-LOOP REVIEW ─────────────────────────────────────────
        PRIORITY_ICON = {
            'Critical': '🔴', 'High': '🟠',
            'Growth': '🟢', 'Opportunity': '🔵', 'Advisory': '⚪',
        }

        def _print_recommendations_table(df, title="Daftar Rekomendasi Kampanye"):
            print("\n" + "═" * 72)
            print(f"  {title}")
            print("═" * 72)
            for row_i, (_, row) in enumerate(df.iterrows()):
                priority = str(row.get('Priority_Level', ''))
                icon = PRIORITY_ICON.get(priority, '⚪')
                channel = row.get('CampaignChannel', '-')
                area = row.get('Decision_Area', '-')
                action = row.get('Recommended_Action', '-')
                timeframe = row.get('Timeframe', '-')
                engagement = row.get('Engagement_Status', '-')
                impact = row.get('Estimated_Impact', '-')
                print(f"\n  [{row_i}] {icon} PRIORITAS: {priority.upper()}")
                print(f"       Kanal Kampanye : {channel}")
                print(f"       Area Keputusan : {area}")
                print(f"       Rekomendasi    : {action}")
                print(f"       Status Pasar   : {engagement}")
                print(f"       Estimasi Dampak: {impact}")
                print(f"       Kapan Dilakukan: {timeframe}")
                print("  " + "─" * 70)
            print(f"\n  Total: {len(df)} rekomendasi\n")

        def _print_summary(rpt):
            lines = rpt.split('\n')
            print("\n" + "─" * 72)
            print("  📋 RINGKASAN HASIL ANALISIS AI")
            print("─" * 72)
            for line in lines:
                line = line.strip()
                if not line or line.startswith('='):
                    continue
                line = line.replace('Priority Distribution:', 'Distribusi Prioritas:')
                line = line.replace('Decision Area Distribution:', 'Distribusi Area Keputusan:')
                line = line.replace('Campaign Channel Focus:', 'Fokus Kanal Kampanye:')
                line = line.replace('TOP RECOMMENDED ACTIONS:', 'AKSI PRIORITAS UTAMA:')
                line = line.replace('MODEL RELIABILITY WARNING:', 'PERINGATAN KEANDALAN MODEL:')
                line = line.replace('recommendation(s)', 'rekomendasi')
                print(f"  {line}")
            print("─" * 72)

        _print_summary(summary_report)

        if recommendations.empty:
            self.recommendations = recommendations
            return True, "No actions to review."

        _print_recommendations_table(recommendations)

        import sys as _sys
        if os.getenv("HITL_AUTO", "0") == "1" or not _sys.stdin.isatty():
            print("\n  ✅ HITL auto/non-interactive aktif — semua rekomendasi otomatis disetujui untuk mode batch/testing.")
            self.reporter.log_hitl_event(
                "Recommendation Review", "auto_approved",
                {"actions": len(recommendations), "mode": "HITL_AUTO"}
            )
            self.recommendations = recommendations
        else:
            while True:
                print("\n┌─────────────────────────────────────────────────────────────────────┐")
                print("│  APA YANG INGIN ANDA LAKUKAN DENGAN REKOMENDASI INI?                 │")
                print("│                                                                       │")
                print("│  [0] ✅  Setuju — jalankan semua rekomendasi                         │")
                print("│  [1] ✏️   Modifikasi — hapus beberapa rekomendasi yang tidak sesuai   │")
                print("│  [2] ❌  Tolak — batalkan semua rekomendasi                          │")
                print("└─────────────────────────────────────────────────────────────────────────┘")

                raw = input("\nMasukkan pilihan Anda (0 / 1 / 2): ").strip()

                if raw == '0':
                    user_input = 'approve'
                elif raw == '1':
                    user_input = 'modify'
                elif raw == '2':
                    user_input = 'reject'
                else:
                    print("\n  ⚠️  Pilihan tidak valid. Masukkan angka 0, 1, atau 2.")
                    continue

                if user_input == 'approve':
                    print("\n  ✅ Semua rekomendasi disetujui dan akan dijalankan.")
                    self.reporter.log_hitl_event(
                        "Recommendation Review", "approved",
                        {"actions": len(recommendations)}
                    )
                    self.recommendations = recommendations
                    break

                elif user_input == 'modify':
                    print("\n" + "─" * 72)
                    print("  ✏️  MODE MODIFIKASI")
                    print("  Anda bisa menghapus rekomendasi yang TIDAK ingin dijalankan.")
                    print("  Rekomendasi yang tersisa akan tetap dieksekusi.")
                    print("─" * 72)
                    _print_recommendations_table(
                        recommendations,
                        "Pilih nomor rekomendasi yang ingin DIHAPUS:"
                    )
                    print("  Contoh: ketik  2,5,7  → menghapus rekomendasi nomor [2], [5], dan [7]")
                    print("  Tekan Enter tanpa mengetik apapun → tidak ada yang dihapus\n")

                    idx_input = input(
                        "  Nomor rekomendasi yang dihapus (pisah koma, atau Enter skip): "
                    ).strip()

                    if idx_input:
                        try:
                            idx_list = [
                                int(i.strip())
                                for i in idx_input.split(',')
                                if i.strip().isdigit()
                            ]
                            invalid = [i for i in idx_list if i >= len(recommendations)]
                            if invalid:
                                print(
                                    f"\n  ⚠️  Nomor tidak valid: {invalid}. "
                                    f"Nomor harus antara 0 dan {len(recommendations)-1}. Coba lagi."
                                )
                                continue

                            mod_recs = recommendations.drop(
                                recommendations.index[idx_list]
                            ).reset_index(drop=True)

                            print(f"\n  ✅ {len(idx_list)} rekomendasi berhasil dihapus.")
                            print(f"  {len(mod_recs)} rekomendasi tersisa akan dijalankan:")
                            _print_recommendations_table(
                                mod_recs,
                                "Rekomendasi yang Tersisa"
                            )
                            self.recommendations = mod_recs
                            self.reporter.log_hitl_event(
                                "Recommendation Review", "modified",
                                {"removed": idx_list, "remaining": len(mod_recs)}
                            )
                            break

                        except Exception as e:
                            print(f"\n  ⚠️  Terjadi kesalahan: {e}. Silakan coba lagi.")
                            continue
                    else:
                        print("\n  ℹ️  Tidak ada yang dihapus. Semua rekomendasi tetap dijalankan.")
                        self.recommendations = recommendations
                        self.reporter.log_hitl_event(
                            "Recommendation Review", "modified_no_change",
                            {"actions": len(recommendations)}
                        )
                        break

                elif user_input == 'reject':
                    print("\n  ❌ Semua rekomendasi DIBATALKAN.")
                    print("  Tidak ada aksi yang akan dijalankan dari hasil analisis ini.")
                    print("  Data analisis tetap tersimpan di folder logs/ untuk referensi.\n")
                    self.reporter.log_hitl_event(
                        "Recommendation Review", "rejected",
                        {"actions_cancelled": len(recommendations)}
                    )
                    self.recommendations = pd.DataFrame()
                    break

        rec_df = getattr(
            self,
            "recommendations",
            pd.DataFrame()
        )

        if not rec_df.empty:
            summary = (
                "Generated digital marketing optimization strategy "
                f"with {len(rec_df)} recommendation(s)."
            )

        else:
            summary = (
                "No additional digital marketing actions were required."
            )

        stats = {
            "Total Recommendations": len(rec_df)
        }

        if not rec_df.empty and 'CustomerID' in rec_df.columns:
            stats["Unique Customers"] = rec_df[
                'CustomerID'
            ].nunique()

        self._stage_snapshot["generate_recommendations"] = {
            "title": "Stage · Digital Marketing Decision Recommendation",
            "summary": summary,
            "stats": stats
        }

        return True, (
            "Digital marketing recommendation completed "
            "with human review."
        )

    def _is_poor_performance(
        self,
        results: Dict[str, Any]
    ) -> bool:
        if self.problem_type == "regression":
            r2 = results.get("r2", -float('inf'))
            return r2 < 0.1

        elif self.problem_type == "classification":
            accuracy = results.get("accuracy", 0)
            return accuracy < 0.6

        return False

    def _performance_metric(
        self,
        results: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[float]]:
        if self.problem_type == "regression":
            r2 = results.get("r2")

            if r2 is not None:
                return "R²", r2

        elif self.problem_type == "classification":
            accuracy = results.get("accuracy")

            if accuracy is not None:
                return "Accuracy", accuracy

        return None, None
