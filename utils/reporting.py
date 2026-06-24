"""
Enhanced Reporting System for Digital Marketing AI Agent MAS.

Provides comprehensive reporting and visualization of workflow results:
- workflow steps
- model performance
- feature analysis
- digital marketing recommendations
- HITL interactions
- publication-ready snapshots
"""

import logging
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import os

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s'
)


class WorkflowReporter:
    """
    Enhanced reporting system for digital marketing AI agent workflow.
    """

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir

        self.report_data = {
            'workflow_start': datetime.now().isoformat(),
            'workflow_goal': None,
            'domain': 'digital_marketing_and_advertising',
            'steps': [],
            'performance_metrics': {},
            'feature_analysis': {},
            'recommendations': {},
            'stage_summaries': [],
            'hitl_events': [],
            'errors': [],
            'summary': {},
            'agentic_evaluation': {},
            'agentic_report_paths': {}
        }

        os.makedirs(output_dir, exist_ok=True)

    # =========================================================
    # STEP LOGGING
    # =========================================================
    def log_step(
        self,
        step_name: str,
        success: bool,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        duration: Optional[float] = None
    ):
        """
        Log a workflow step with enhanced details.
        """

        step_data = {
            'step_name': step_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'details': details or {}
        }

        self.report_data['steps'].append(step_data)

        status_icon = "✅" if success else "❌"
        duration_str = f" ({duration:.2f}s)" if duration else ""

        logging.info(
            f"{status_icon} {step_name}: {message}{duration_str}"
        )

        if details:
            for key, value in details.items():
                if isinstance(value, dict):
                    logging.info(
                        f"   📊 {key}: {json.dumps(value, indent=2)}"
                    )
                else:
                    logging.info(
                        f"   📊 {key}: {value}"
                    )

    # =========================================================
    # STAGE SUMMARY
    # =========================================================
    def log_stage_summary(
        self,
        stage_name: str,
        summary: str,
        stats: Optional[Dict[str, Any]] = None
    ):
        """
        Log a structured stage summary.
        """

        stats = stats or {}

        self.report_data['stage_summaries'].append({
            'stage_name': stage_name,
            'summary': summary,
            'stats': stats
        })

        logging.info("")
        logging.info(f"--- {stage_name} ---")
        logging.info(summary)

        for key, value in stats.items():
            logging.info(
                f"   • {key}: {value}"
            )

    # =========================================================
    # HITL LOGGING
    # =========================================================
    def log_hitl_event(
        self,
        title: str,
        decision: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a human-in-the-loop interaction.
        """

        metadata = metadata or {}

        event_entry = {
            "timestamp": datetime.now().isoformat(),
            "title": title,
            "decision": decision,
            "metadata": metadata
        }

        self.report_data['hitl_events'].append(event_entry)

        context_bits = (
            ", ".join(
                f"{k}={v}"
                for k, v in metadata.items()
            )
            if metadata
            else ""
        )

        suffix = f" ({context_bits})" if context_bits else ""

        logging.info(
            f"👤 HITL · {title}: {decision}{suffix}"
        )

    # =========================================================
    # PERFORMANCE METRICS
    # =========================================================
    def log_performance_metrics(
        self,
        metrics: Dict[str, Any]
    ):
        """
        Log model performance metrics.
        """

        self.report_data['performance_metrics'] = metrics

        logging.info("🎯 MODEL PERFORMANCE METRICS")
        logging.info("=" * 50)

        for metric_name, value in metrics.items():
            if isinstance(value, float):
                if (
                    'accuracy' in metric_name.lower()
                    or 'r2' in metric_name.lower()
                    or 'f1' in metric_name.lower()
                    or 'precision' in metric_name.lower()
                    or 'recall' in metric_name.lower()
                    or 'auc' in metric_name.lower()
                ):
                    logging.info(
                        f"   {metric_name}: "
                        f"{value:.4f} ({value * 100:.2f}%)"
                    )
                else:
                    logging.info(
                        f"   {metric_name}: {value:.4f}"
                    )
            else:
                logging.info(
                    f"   {metric_name}: {value}"
                )

    # =========================================================
    # FEATURE ANALYSIS
    # =========================================================
    def log_feature_analysis(
        self,
        analysis: Dict[str, Any]
    ):
        """
        Log intelligent feature analysis results.
        """

        self.report_data['feature_analysis'] = analysis

        logging.info("🧠 INTELLIGENT FEATURE ANALYSIS")
        logging.info("=" * 50)

        try:
            if 'summary' in analysis:
                logging.info(analysis['summary'])

            if 'recommendations' in analysis:
                recs = analysis['recommendations']

                if (
                    'features_to_remove' in recs
                    and recs['features_to_remove']
                ):
                    logging.info("\n🗑️ Features Removed:")

                    for feature in recs['features_to_remove']:
                        feature_name = feature.get(
                            'feature',
                            'Unknown'
                        )

                        reason = feature.get(
                            'reason',
                            'No reason provided'
                        )

                        logging.info(
                            f"   • {feature_name}: {reason}"
                        )

                if (
                    'features_to_keep' in recs
                    and recs['features_to_keep']
                ):
                    logging.info("\n⭐ Top Marketing Features to Keep:")

                    for feature in recs['features_to_keep'][:5]:
                        feature_name = feature.get(
                            'feature',
                            'Unknown'
                        )

                        score = feature.get(
                            'score',
                            0.0
                        )

                        logging.info(
                            f"   • {feature_name}: {score:.3f}"
                        )

        except Exception as e:
            logging.warning(
                f"Error logging feature analysis: {e}"
            )

            logging.info(
                "Feature analysis completed but detailed logging failed."
            )

    # =========================================================
    # RECOMMENDATION LOGGING
    # =========================================================
    def log_recommendations(
        self,
        recommendations: Dict[str, Any]
    ):
        """
        Log digital marketing recommendations.
        """

        self.report_data['recommendations'] = recommendations

        logging.info("🎯 DIGITAL MARKETING DECISION RECOMMENDATIONS")
        logging.info("=" * 60)

        if 'summary_report' in recommendations:
            logging.info(recommendations['summary_report'])

        if 'recommendations' in recommendations:
            recs = recommendations['recommendations']

            if hasattr(recs, 'empty'):
                if not recs.empty:
                    logging.info("\n📋 Detailed Marketing Recommendations:")

                    for i, (_, rec) in enumerate(recs.iterrows(), 1):
                        self._log_single_recommendation(i, rec)

            elif recs:
                logging.info("\n📋 Detailed Marketing Recommendations:")

                for i, rec in enumerate(recs, 1):
                    self._log_single_recommendation(i, rec)

    # =========================================================
    # ERROR LOGGING
    # =========================================================
    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Log errors with context.
        """

        error_data = {
            'error_type': error_type,
            'error_message': error_message,
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        }

        self.report_data['errors'].append(error_data)

        logging.error(
            f"❌ {error_type}: {error_message}"
        )

        if context:
            for key, value in context.items():
                logging.error(
                    f"   Context: {key} = {value}"
                )

    # =========================================================
    # PUBLICATION SNAPSHOT
    # =========================================================
    def save_publication_snapshot(
        self,
        filename_prefix: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        """
        Save publication-ready workflow snapshot.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = filename_prefix or f"publication_snapshot_{timestamp}"

        json_path = os.path.join(
            self.output_dir,
            f"{prefix}.json"
        )

        csv_path = None

        snapshot = {
            "generated_at": datetime.now().isoformat(),
            "domain": "digital_marketing_and_advertising",
            "stage_summaries": self.report_data.get(
                'stage_summaries',
                []
            ),
            "performance_metrics": self.report_data.get(
                'performance_metrics',
                {}
            ),
            "workflow_summary": self.report_data.get(
                'summary',
                {}
            ),
            "agentic_evaluation": self.report_data.get(
                'agentic_evaluation',
                {}
            ),
            "agentic_report_paths": self.report_data.get(
                'agentic_report_paths',
                {}
            )
        }

        recommendations_obj = self.report_data.get(
            'recommendations',
            {}
        )

        recs = recommendations_obj.get('recommendations')

        if hasattr(recs, 'empty') and not recs.empty:
            preferred_columns = [
                'CustomerID',
                'Predicted_Conversion',
                'Predicted_Value',
                'Priority_Level',
                'Decision_Area',
                'Recommended_Action',
                'Estimated_Impact',
                'Timeframe',
                'CampaignChannel',
                'CampaignType',
                'Engagement_Status',
                'Contributing_Factors',
                'Reason_for_Action',
                'Model_Confidence',
                'Model_Warning'
            ]

            available_cols = [
                col for col in preferred_columns
                if col in recs.columns
            ]

            trimmed_recs = (
                recs[available_cols].copy()
                if available_cols
                else recs.copy()
            )

            snapshot['recommendations'] = trimmed_recs.head(
                25
            ).to_dict(orient='records')

            csv_path = os.path.join(
                self.output_dir,
                f"{prefix}_recommendations.csv"
            )

            trimmed_recs.to_csv(
                csv_path,
                index=False
            )

        elif isinstance(recs, list) and recs:
            snapshot['recommendations'] = recs[:25]

        else:
            snapshot['recommendations'] = []

        with open(json_path, 'w') as f:
            json.dump(
                snapshot,
                f,
                indent=2,
                default=str
            )

        logging.info(
            f"📄 Publication snapshot saved to: {json_path}"
        )

        if csv_path:
            logging.info(
                f"📊 Recommendation preview saved to: {csv_path}"
            )

        return {
            "json": json_path,
            "csv": csv_path
        }

    # =========================================================
    # SINGLE RECOMMENDATION FORMATTER
    # =========================================================
    def _log_single_recommendation(
        self,
        index: int,
        rec: Dict[str, Any]
    ):
        """
        Format a single digital marketing recommendation line.
        """

        def _maybe_value(*keys):
            for key in keys:
                if key in rec and rec[key] is not None:
                    val = rec[key]

                    if isinstance(val, float) and pd.isna(val):
                        continue

                    return val

            return None

        customer_id = _maybe_value(
            'CustomerID',
            'Customer_ID',
            'customer_id'
        )

        action_value = _maybe_value(
            'Recommended_Action',
            'Action'
        )

        decision_area = _maybe_value(
            'Decision_Area'
        )

        if action_value and customer_id:
            action_text = (
                f"Customer {customer_id}: {action_value}"
            )

        elif action_value:
            action_text = str(action_value)

        elif customer_id:
            action_text = f"Customer {customer_id}"

        else:
            action_text = "Recommendation unavailable"

        logging.info(f"\n   {index}. {action_text}")

        if decision_area:
            logging.info(
                f"      Decision Area: {decision_area}"
            )

        priority = _maybe_value(
            'Priority_Level',
            'Priority'
        )

        if (
            priority
            and str(priority).strip().lower()
            not in {'n/a', 'unknown', 'none', ''}
        ):
            logging.info(
                f"      Priority: {priority}"
            )

        channel = _maybe_value(
            'CampaignChannel',
            'Campaign_Channel'
        )

        if channel:
            logging.info(
                f"      Campaign Channel: {channel}"
            )

        campaign_type = _maybe_value(
            'CampaignType',
            'Campaign_Type'
        )

        if campaign_type:
            logging.info(
                f"      Campaign Type: {campaign_type}"
            )

        predicted_conversion = _maybe_value(
            'Predicted_Conversion'
        )

        if predicted_conversion is not None:
            logging.info(
                f"      Predicted Conversion: {predicted_conversion}"
            )

        predicted_value = _maybe_value(
            'Predicted_Value'
        )

        if predicted_value is not None:
            logging.info(
                f"      Predicted Value: {predicted_value}"
            )

        estimated_impact = _maybe_value(
            'Estimated_Impact'
        )

        if estimated_impact:
            logging.info(
                f"      Estimated Impact: {estimated_impact}"
            )

        confidence = _maybe_value(
            'Model_Confidence',
            'Confidence'
        )

        if (
            confidence
            and str(confidence).strip().lower()
            not in {'n/a', 'unknown', 'none', ''}
        ):
            logging.info(
                f"      Confidence: {confidence}"
            )

        engagement = _maybe_value(
            'Engagement_Status'
        )

        if engagement:
            logging.info(
                f"      Engagement Status: {engagement}"
            )

        factors = _maybe_value(
            'Contributing_Factors',
            'Top_Indicators'
        )

        if factors and str(factors).strip():
            logging.info(
                f"      Factors: {factors}"
            )

        timeframe = _maybe_value(
            'Timeframe'
        )

        if (
            timeframe
            and str(timeframe).strip().lower()
            not in {'n/a', 'unknown', 'none', ''}
        ):
            logging.info(
                f"      Timeframe: {timeframe}"
            )

        reason = _maybe_value(
            'Reason_for_Action'
        )

        if reason and str(reason).strip():
            logging.info(
                f"      Reason: {reason}"
            )

    # =========================================================
    # SUMMARY GENERATION
    # =========================================================
    def generate_summary(self):
        """
        Generate comprehensive workflow summary.
        """

        total_steps = len(self.report_data['steps'])

        successful_steps = sum(
            1 for step in self.report_data['steps']
            if step['success']
        )

        failed_steps = total_steps - successful_steps

        start_time = datetime.fromisoformat(
            self.report_data['workflow_start']
        )

        end_time = datetime.now()

        total_duration = (
            end_time - start_time
        ).total_seconds()

        performance_summary = {}

        if self.report_data['performance_metrics']:
            metrics = self.report_data['performance_metrics']

            for metric_name in [
                'accuracy',
                'precision',
                'recall',
                'f1_score',
                'roc_auc',
                'r2',
                'mse',
                'rmse',
                'model_used'
            ]:
                if metric_name in metrics:
                    performance_summary[metric_name] = metrics[
                        metric_name
                    ]

        feature_summary = {}

        if self.report_data['feature_analysis']:
            analysis = self.report_data['feature_analysis']

            if 'recommendations' in analysis:
                recs = analysis['recommendations']

                feature_summary['features_removed'] = len(
                    recs.get('features_to_remove', [])
                )

                feature_summary['features_kept'] = len(
                    recs.get('features_to_keep', [])
                )

        rec_summary = {}

        if self.report_data['recommendations']:
            recs = self.report_data['recommendations']

            if 'recommendations' in recs:
                recommendations_list = recs['recommendations']

                if hasattr(recommendations_list, 'iterrows'):
                    rec_summary['total_recommendations'] = len(
                        recommendations_list
                    )

                    priority_counts = {}

                    for _, row in recommendations_list.iterrows():
                        priority = row.get(
                            'Priority_Level',
                            'Unknown'
                        )

                        priority_counts[priority] = (
                            priority_counts.get(priority, 0) + 1
                        )

                    rec_summary[
                        'priority_distribution'
                    ] = priority_counts

                    if 'Decision_Area' in recommendations_list.columns:
                        rec_summary[
                            'decision_area_distribution'
                        ] = recommendations_list[
                            'Decision_Area'
                        ].value_counts().to_dict()

                    if 'CampaignChannel' in recommendations_list.columns:
                        rec_summary[
                            'channel_distribution'
                        ] = recommendations_list[
                            'CampaignChannel'
                        ].value_counts().to_dict()

                elif isinstance(recommendations_list, list):
                    rec_summary['total_recommendations'] = len(
                        recommendations_list
                    )

                    priority_counts = {}

                    for rec in recommendations_list:
                        priority = rec.get(
                            'Priority_Level',
                            'Unknown'
                        )

                        priority_counts[priority] = (
                            priority_counts.get(priority, 0) + 1
                        )

                    rec_summary[
                        'priority_distribution'
                    ] = priority_counts

        self.report_data['summary'] = {
            'workflow_duration_seconds': total_duration,
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'success_rate': (
                successful_steps / total_steps
                if total_steps > 0
                else 0
            ),
            'performance_metrics': performance_summary,
            'feature_analysis': feature_summary,
            'recommendations': rec_summary,
            'workflow_end': end_time.isoformat()
        }

        return self.report_data['summary']

    # =========================================================
    # FINAL SUMMARY PRINT
    # =========================================================
    def print_final_summary(self):
        """
        Print comprehensive final summary to console.
        """

        summary = self.generate_summary()

        logging.info("\n" + "=" * 80)
        logging.info("🎉 DIGITAL MARKETING AI AGENT WORKFLOW RECAP")
        logging.info("=" * 80)

        logging.info(
            f"🗂️  Goal: {self.report_data.get('workflow_goal', 'N/A')}"
        )

        logging.info(
            f"📣 Domain: Digital Marketing & Advertising"
        )

        logging.info(
            f"⏱️  Duration: "
            f"{summary['workflow_duration_seconds']:.2f}s"
        )

        logging.info(
            f"📊 Steps: "
            f"{summary['successful_steps']}/"
            f"{summary['total_steps']} succeeded "
            f"({summary['success_rate'] * 100:.1f}%)"
        )

        if summary['performance_metrics']:
            logging.info("\n🎯 MODEL PERFORMANCE SNAPSHOT")

            for metric, value in summary[
                'performance_metrics'
            ].items():
                if isinstance(value, float):
                    if (
                        'accuracy' in metric.lower()
                        or 'r2' in metric.lower()
                        or 'precision' in metric.lower()
                        or 'recall' in metric.lower()
                        or 'f1' in metric.lower()
                        or 'auc' in metric.lower()
                    ):
                        logging.info(
                            f"   {metric}: "
                            f"{value:.4f} ({value * 100:.2f}%)"
                        )
                    else:
                        logging.info(
                            f"   {metric}: {value:.4f}"
                        )
                else:
                    logging.info(
                        f"   {metric}: {value}"
                    )

        if summary['feature_analysis']:
            fa = summary['feature_analysis']

            logging.info("\n🧠 FEATURE ANALYSIS RECAP")

            logging.info(
                f"   Features Removed: "
                f"{fa.get('features_removed', 0)}"
            )

            logging.info(
                f"   Features Kept: "
                f"{fa.get('features_kept', 0)}"
            )

        if summary['recommendations']:
            rec = summary['recommendations']

            logging.info("\n📋 DIGITAL MARKETING RECOMMENDATIONS")

            logging.info(
                f"   Total Recommendations: "
                f"{rec.get('total_recommendations', 0)}"
            )

            if 'priority_distribution' in rec:
                logging.info("   Priority Distribution:")

                for priority, count in rec[
                    'priority_distribution'
                ].items():
                    logging.info(
                        f"     {priority}: {count}"
                    )

            if 'decision_area_distribution' in rec:
                logging.info("   Decision Area Distribution:")

                for area, count in rec[
                    'decision_area_distribution'
                ].items():
                    logging.info(
                        f"     {area}: {count}"
                    )

            if 'channel_distribution' in rec:
                logging.info("   Campaign Channel Distribution:")

                for channel, count in rec[
                    'channel_distribution'
                ].items():
                    logging.info(
                        f"     {channel}: {count}"
                    )

        if self.report_data['errors']:
            logging.info(
                f"\n❌ ERRORS ENCOUNTERED "
                f"({len(self.report_data['errors'])})"
            )

            for error in self.report_data['errors']:
                logging.info(
                    f"   • {error['error_type']}: "
                    f"{error['error_message']}"
                )

        else:
            logging.info("\n✅ No errors recorded.")

        logging.info("\n--- Stage Highlights ---")

        for entry in self.report_data.get(
            'stage_summaries',
            []
        ):
            logging.info(
                f"• {entry['stage_name']}: {entry['summary']}"
            )

        if self.report_data.get('hitl_events'):
            logging.info("\n👤 HITL INTERACTIONS")

            for event in self.report_data['hitl_events']:
                meta = event.get("metadata") or {}

                meta_text = ", ".join(
                    f"{k}={v}"
                    for k, v in meta.items()
                )

                logging.info(
                    f"   {event['title']} → "
                    f"{event['decision']} ({meta_text})"
                )

        logging.info("\n" + "=" * 80 + "\n")

    # =========================================================
    # SAVE REPORT
    # =========================================================
    def save_report(
        self,
        filename: Optional[str] = None
    ):
        """
        Save complete report to JSON file.
        """

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"digital_marketing_workflow_report_{timestamp}.json"

        filepath = os.path.join(
            self.output_dir,
            filename
        )

        with open(filepath, 'w') as f:
            json.dump(
                self.report_data,
                f,
                indent=2,
                default=str
            )

        logging.info(
            f"📄 Complete digital marketing report saved to: {filepath}"
        )

        return filepath


def create_reporter(
    output_dir: str = "logs"
) -> WorkflowReporter:
    """
    Factory function to create a workflow reporter.
    """

    return WorkflowReporter(output_dir)


if __name__ == "__main__":
    reporter = create_reporter()

    reporter.log_step(
        "Data Loading",
        True,
        "Digital marketing campaign dataset loaded successfully",
        {
            "shape": (8000, 20),
            "missing_values": 0
        },
        2.5
    )

    reporter.log_step(
        "Preprocessing",
        True,
        "Campaign and customer features preprocessed",
        {
            "features_removed": 2,
            "features_kept": 17
        },
        1.8
    )

    reporter.log_performance_metrics({
        "accuracy": 0.92,
        "precision": 0.90,
        "recall": 0.91,
        "f1_score": 0.905
    })

    reporter.log_feature_analysis({
        "summary": "Marketing feature analysis completed",
        "recommendations": {
            "features_to_remove": [
                {
                    "feature": "CustomerID",
                    "reason": "identifier"
                }
            ],
            "features_to_keep": [
                {
                    "feature": "ClickThroughRate",
                    "score": 0.8
                },
                {
                    "feature": "ConversionRate",
                    "score": 0.75
                }
            ]
        }
    })

    sample_recs = pd.DataFrame([
        {
            "CustomerID": 8000,
            "Predicted_Conversion": 0,
            "Priority_Level": "Critical",
            "Decision_Area": "Creative & Targeting Optimization",
            "Recommended_Action": "Revise ad creative and targeting.",
            "Estimated_Impact": "High",
            "Timeframe": "Immediate",
            "CampaignChannel": "Social Media",
            "CampaignType": "Awareness",
            "Engagement_Status": "low CTR",
            "Contributing_Factors": "ClickThroughRate=0.043",
            "Reason_for_Action": "Predicted low conversion with weak CTR.",
            "Model_Confidence": "High"
        }
    ])

    reporter.log_recommendations({
        "recommendations": sample_recs,
        "summary_report": (
            "Digital marketing decision recommendation generated."
        )
    })

    reporter.print_final_summary()
    reporter.save_report()