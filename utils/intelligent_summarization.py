import logging
import json
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
import os

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s'
)


class IntelligentSummarizer:
    """
    LLM-powered intelligent summarization for Digital Marketing AI Agent MAS.

    Stores detailed workflow results and generates clean summaries for:
    - dataset analysis
    - preprocessing
    - model performance
    - feature analysis
    - advertising decision recommendations
    """

    def __init__(self, llm_agent=None):
        self.llm_agent = llm_agent

        self.stored_results = {
            'workflow_start_time': None,
            'workflow_end_time': None,
            'domain': 'digital_marketing_and_advertising',
            'dataset_info': {},
            'preprocessing_steps': [],
            'model_results': [],
            'feature_analysis': {},
            'recommendations': {},
            'performance_metrics': {},
            'adaptive_intelligence_events': [],
            'errors': [],
            'summary': None
        }

        self.logging_enabled = True

    # =========================================================
    # STORE WORKFLOW DATA
    # =========================================================
    def store_workflow_start(
        self,
        dataset_path: str,
        problem_type: str,
        target_column: str,
        feature_columns: List[str]
    ):
        self.stored_results['workflow_start_time'] = (
            datetime.now().isoformat()
        )

        self.stored_results['dataset_info'] = {
            'dataset_path': dataset_path,
            'problem_type': problem_type,
            'target_column': target_column,
            'feature_columns': feature_columns
        }

        self._log_info(
            "🚀 Digital Marketing Workflow started",
            f"Dataset: {os.path.basename(dataset_path)}"
        )

    def store_preprocessing_step(
        self,
        step_name: str,
        details: Dict[str, Any],
        duration: float = None
    ):
        step_data = {
            'timestamp': datetime.now().isoformat(),
            'step_name': step_name,
            'details': details,
            'duration': duration
        }

        self.stored_results['preprocessing_steps'].append(step_data)

        self._log_info(
            f"🔧 Preprocessing: {step_name}",
            f"Duration: {duration:.2f}s" if duration else ""
        )

    def store_model_result(
        self,
        model_name: str,
        performance: Dict[str, Any],
        adaptive_intelligence: bool = False,
        tried_models: List[str] = None
    ):
        model_data = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'performance': performance,
            'adaptive_intelligence_used': adaptive_intelligence,
            'tried_models': tried_models or []
        }

        self.stored_results['model_results'].append(model_data)

        if adaptive_intelligence and tried_models:
            self.stored_results[
                'adaptive_intelligence_events'
            ].append({
                'timestamp': datetime.now().isoformat(),
                'trigger_model': model_name,
                'tried_models': tried_models,
                'final_model': model_name,
                'performance_improvement': performance
            })

        if self.logging_enabled:
            if adaptive_intelligence:
                self._log_info(
                    f"🧠 Adaptive Intelligence: {model_name}",
                    f"Performance: {self._format_performance(performance)}"
                )

            else:
                self._log_info(
                    f"🤖 Model: {model_name}",
                    f"Performance: {self._format_performance(performance)}"
                )

    def store_feature_analysis(
        self,
        analysis: Dict[str, Any]
    ):
        self.stored_results['feature_analysis'] = analysis

        self._log_info(
            "🧠 Marketing Feature Analysis",
            "Completed intelligent feature analysis"
        )

    def store_recommendations(
        self,
        recommendations: Dict[str, Any]
    ):
        self.stored_results['recommendations'] = recommendations

        self._log_info(
            "🎯 Digital Marketing Recommendations",
            "Generated advertising decision recommendations"
        )

    def store_error(
        self,
        error_type: str,
        error_message: str,
        context: Dict[str, Any] = None
    ):
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'message': error_message,
            'context': context or {}
        }

        self.stored_results['errors'].append(error_data)

        self._log_error(
            f"❌ {error_type}",
            error_message
        )

    def store_workflow_end(self):
        self.stored_results['workflow_end_time'] = (
            datetime.now().isoformat()
        )

        self._log_info(
            "✅ Digital Marketing Workflow Complete",
            "Workflow execution finished"
        )

    # =========================================================
    # SUMMARY GENERATION
    # =========================================================
    def generate_intelligent_summary(self) -> str:
        """
        Generate an intelligent summary using LLM if available.
        """

        if not self.llm_agent:
            return self._generate_basic_summary()

        try:
            summary_prompt = self._build_summary_prompt()

            response = self.llm_agent.generate(
                summary_prompt,
                max_tokens=1000
            )

            parsed = response.get('parsed')

            if parsed and 'summary' in parsed:
                self.stored_results['summary'] = parsed['summary']
                return parsed['summary']

            if response and 'summary' in response:
                self.stored_results['summary'] = response['summary']
                return response['summary']

            raw = response.get('raw') if response else None

            if raw:
                self.stored_results['summary'] = raw
                return raw

            return self._generate_basic_summary()

        except Exception as e:
            logging.warning(
                f"LLM summarization failed: {e}"
            )

            return self._generate_basic_summary()

    def _build_summary_prompt(self) -> str:
        """
        Build a prompt for LLM-based marketing workflow summarization.
        """

        workflow_data = self.stored_results

        prompt_parts = [
            "You are an expert data scientist specializing in digital marketing analytics and advertising decision support.",
            "",
            "Provide a professional technical summary of the AI agent workflow.",
            "Focus on campaign performance, conversion prediction, feature drivers, and advertising recommendations.",
            "",
            "WORKFLOW INFORMATION:",
            f"Domain: Digital Marketing and Advertising",
            f"Dataset: {workflow_data['dataset_info'].get('dataset_path', 'Unknown')}",
            f"Problem Type: {workflow_data['dataset_info'].get('problem_type', 'Unknown')}",
            f"Target Column: {workflow_data['dataset_info'].get('target_column', 'Unknown')}",
            f"Features: {len(workflow_data['dataset_info'].get('feature_columns', []))} columns",
            "",
            "MODEL PERFORMANCE ANALYSIS:"
        ]

        for i, model_result in enumerate(
            workflow_data['model_results'],
            1
        ):
            model_name = model_result['model_name']
            performance = model_result['performance']
            adaptive = model_result['adaptive_intelligence_used']
            tried_models = model_result.get('tried_models', [])

            prompt_parts.append(f"Model {i}: {model_name}")

            if 'accuracy' in performance:
                acc_val = performance['accuracy']

                prompt_parts.append(
                    f"  Accuracy: {acc_val:.4f} "
                    f"({'EXCELLENT' if acc_val > 0.9 else 'GOOD' if acc_val > 0.7 else 'POOR' if acc_val > 0.5 else 'FAILED'})"
                )

            if 'precision' in performance:
                prompt_parts.append(
                    f"  Precision: {performance['precision']:.4f}"
                )

            if 'recall' in performance:
                prompt_parts.append(
                    f"  Recall: {performance['recall']:.4f}"
                )

            if 'f1_score' in performance:
                prompt_parts.append(
                    f"  F1 Score: {performance['f1_score']:.4f}"
                )

            if 'roc_auc' in performance and performance['roc_auc'] is not None:
                prompt_parts.append(
                    f"  ROC-AUC: {performance['roc_auc']:.4f}"
                )

            if 'r2' in performance:
                r2_val = performance['r2']

                if r2_val == float('-inf'):
                    prompt_parts.append(
                        "  R² Score: FAILED"
                    )

                else:
                    prompt_parts.append(
                        f"  R² Score: {r2_val:.4f} "
                        f"({'EXCELLENT' if r2_val > 0.8 else 'GOOD' if r2_val > 0.5 else 'POOR' if r2_val > 0.1 else 'FAILED'})"
                    )

            if 'mse' in performance:
                prompt_parts.append(
                    f"  MSE: {performance['mse']:.4f}"
                )

            if 'rmse' in performance:
                prompt_parts.append(
                    f"  RMSE: {performance['rmse']:.4f}"
                )

            if adaptive:
                prompt_parts.append(
                    "  Adaptive Intelligence Activated"
                )

                prompt_parts.append(
                    f"  Tried Models: "
                    f"{', '.join(tried_models) if tried_models else 'None'}"
                )

            prompt_parts.append("")

        if workflow_data['feature_analysis']:
            recs = workflow_data['feature_analysis'].get(
                'recommendations',
                {}
            )

            prompt_parts.extend([
                "FEATURE ANALYSIS:",
                f"Features Kept: {len(recs.get('features_to_keep', []))}",
                f"Features Removed: {len(recs.get('features_to_remove', []))}",
                ""
            ])

        if workflow_data['recommendations']:
            recs = workflow_data['recommendations'].get(
                'recommendations',
                []
            )

            if hasattr(recs, 'empty'):
                rec_count = len(recs) if not recs.empty else 0

            else:
                rec_count = len(recs) if recs else 0

            prompt_parts.extend([
                "DIGITAL MARKETING RECOMMENDATIONS:",
                f"Total Recommendations: {rec_count}",
                ""
            ])

        if workflow_data['errors']:
            prompt_parts.extend([
                "ERRORS:",
                f"Total Errors: {len(workflow_data['errors'])}",
                ""
            ])

        prompt_parts.extend([
            "SUMMARY REQUIREMENTS:",
            "Write a concise but technical report covering:",
            "",
            "1. Workflow status and whether the AI agent completed the analysis.",
            "2. Dataset and target variable used for marketing prediction.",
            "3. Model performance and reliability.",
            "4. Most important marketing/campaign features if available.",
            "5. Digital marketing decision recommendations generated.",
            "6. Any limitations, warnings, or next improvement steps.",
            "",
            "Use digital marketing language such as conversion prediction, campaign optimization, audience targeting, budget allocation, engagement, CTR, conversion rate, retention, and channel performance.",
            "",
            "Return ONLY valid JSON:",
            "{\"summary\":\"your detailed summary here\"}"
        ])

        return "\n".join(prompt_parts)

    def _generate_basic_summary(self) -> str:
        """
        Generate a basic summary without LLM.
        """

        workflow_data = self.stored_results

        summary_lines = [
            "=" * 60,
            "🎯 DIGITAL MARKETING AI AGENT WORKFLOW SUMMARY",
            "=" * 60,
            f"📊 Dataset: {os.path.basename(workflow_data['dataset_info'].get('dataset_path', 'Unknown'))}",
            f"📣 Domain: Digital Marketing & Advertising",
            f"🎯 Problem Type: {workflow_data['dataset_info'].get('problem_type', 'Unknown')}",
            f"🏷️ Target Column: {workflow_data['dataset_info'].get('target_column', 'Unknown')}",
            f"📈 Features: {len(workflow_data['dataset_info'].get('feature_columns', []))} columns",
            ""
        ]

        if workflow_data['model_results']:
            summary_lines.append("🤖 MODEL PERFORMANCE:")

            for i, model_result in enumerate(
                workflow_data['model_results'],
                1
            ):
                model_name = model_result['model_name']
                performance = model_result['performance']
                adaptive = model_result['adaptive_intelligence_used']

                summary_lines.append(
                    f"  {i}. {model_name}"
                )

                if 'accuracy' in performance:
                    summary_lines.append(
                        f"     Accuracy: {performance['accuracy']:.4f} "
                        f"({performance['accuracy'] * 100:.1f}%)"
                    )

                if 'precision' in performance:
                    summary_lines.append(
                        f"     Precision: {performance['precision']:.4f}"
                    )

                if 'recall' in performance:
                    summary_lines.append(
                        f"     Recall: {performance['recall']:.4f}"
                    )

                if 'f1_score' in performance:
                    summary_lines.append(
                        f"     F1 Score: {performance['f1_score']:.4f}"
                    )

                if (
                    'roc_auc' in performance
                    and performance['roc_auc'] is not None
                ):
                    summary_lines.append(
                        f"     ROC-AUC: {performance['roc_auc']:.4f}"
                    )

                if 'r2' in performance:
                    summary_lines.append(
                        f"     R² Score: {performance['r2']:.4f} "
                        f"({performance['r2'] * 100:.1f}%)"
                    )

                if 'mse' in performance:
                    summary_lines.append(
                        f"     MSE: {performance['mse']:.4f}"
                    )

                if 'rmse' in performance:
                    summary_lines.append(
                        f"     RMSE: {performance['rmse']:.4f}"
                    )

                if adaptive:
                    summary_lines.append(
                        f"     🧠 Adaptive Intelligence: Tried "
                        f"{len(model_result.get('tried_models', []))} models"
                    )

                summary_lines.append("")

        if workflow_data['feature_analysis']:
            recs = workflow_data['feature_analysis'].get(
                'recommendations',
                {}
            )

            summary_lines.extend([
                "🧠 MARKETING FEATURE ANALYSIS:",
                f"  Features Kept: {len(recs.get('features_to_keep', []))}",
                f"  Features Removed: {len(recs.get('features_to_remove', []))}",
                ""
            ])

        if workflow_data['recommendations']:
            recs = workflow_data['recommendations'].get(
                'recommendations',
                []
            )

            if hasattr(recs, 'empty'):
                rec_count = len(recs) if not recs.empty else 0

            else:
                rec_count = len(recs) if recs else 0

            summary_lines.extend([
                "🎯 DIGITAL MARKETING RECOMMENDATIONS:",
                f"  Total Generated: {rec_count}",
                ""
            ])

        if workflow_data['errors']:
            summary_lines.extend([
                "⚠️ ISSUES:",
                f"  Errors Encountered: {len(workflow_data['errors'])}",
                ""
            ])

        if (
            workflow_data['workflow_start_time']
            and workflow_data['workflow_end_time']
        ):
            start_time = datetime.fromisoformat(
                workflow_data['workflow_start_time']
            )

            end_time = datetime.fromisoformat(
                workflow_data['workflow_end_time']
            )

            duration = (
                end_time - start_time
            ).total_seconds()

            summary_lines.append(
                f"⏱️ Total Duration: {duration:.2f} seconds"
            )

        summary_lines.append("=" * 60)

        return "\n".join(summary_lines)

    # =========================================================
    # FORMAT HELPERS
    # =========================================================
    def _format_performance(
        self,
        performance: Dict[str, Any]
    ) -> str:
        if 'accuracy' in performance:
            return f"Accuracy: {performance['accuracy']:.4f}"

        if 'f1_score' in performance:
            return f"F1 Score: {performance['f1_score']:.4f}"

        if 'r2' in performance:
            return f"R²: {performance['r2']:.4f}"

        if 'mse' in performance:
            return f"MSE: {performance['mse']:.4f}"

        return "Performance metrics available"

    def _log_info(
        self,
        title: str,
        message: str = ""
    ):
        if self.logging_enabled:
            if message:
                logging.info(
                    f"[Summarizer] {title}: {message}"
                )

            else:
                logging.info(
                    f"[Summarizer] {title}"
                )

    def _log_error(
        self,
        title: str,
        message: str
    ):
        if self.logging_enabled:
            logging.error(
                f"[Summarizer] {title}: {message}"
            )

    # =========================================================
    # SAVE RESULTS
    # =========================================================
    def save_detailed_results(
        self,
        filepath: str = None
    ) -> str:
        """
        Save detailed results to JSON file.
        """

        if not filepath:
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            filepath = (
                f"logs/digital_marketing_detailed_results_{timestamp}.json"
            )

        os.makedirs(
            os.path.dirname(filepath),
            exist_ok=True
        )

        with open(filepath, 'w') as f:
            json.dump(
                self.stored_results,
                f,
                indent=4,
                default=str
            )

        return filepath

    # =========================================================
    # LOGGING MODE
    # =========================================================
    def disable_logging(self):
        self.logging_enabled = False

    def enable_logging(self):
        self.logging_enabled = True

    def set_logging_mode(
        self,
        mode: str = "verbose"
    ):
        if mode == "verbose":
            self.logging_enabled = True

        elif mode == "minimal":
            self.logging_enabled = True

        elif mode == "silent":
            self.logging_enabled = False

        else:
            self.logging_enabled = True


def create_summarizer(
    llm_agent=None
) -> IntelligentSummarizer:
    """
    Factory function to create an IntelligentSummarizer instance.
    """

    return IntelligentSummarizer(llm_agent)