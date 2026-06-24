import pandas as pd
import logging
from typing import Dict, Any
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s'
)


class OptimizationAgent:
    """
    OptimizationAgent takes model analysis results and generates
    digital marketing and advertising decision recommendations.

    This is the final step of the MAS workflow:
    prediction -> insight -> decision recommendation.
    """

    def __init__(self, analysis_results: Dict[str, Any]):
        logging.info("Initializing Digital Marketing Optimization Agent...")

        self.results = analysis_results

        if 'results_df' in analysis_results:
            if (
                'results_df' not in analysis_results
                or 'anomaly_labels' not in analysis_results
            ):
                raise ValueError(
                    "Anomaly detection results missing required keys: "
                    "'results_df' and 'anomaly_labels'"
                )

        else:
            required_keys = [
                'test_data',
                'test_predictions',
                'feature_importances'
            ]

            if not all(k in self.results for k in required_keys):
                raise ValueError(
                    "Analysis results are missing required keys "
                    "for marketing optimization."
                )

    # =========================================================
    # MODEL PERFORMANCE
    # =========================================================
    def _assess_model_performance(self) -> Dict[str, Any]:
        """
        Assess model performance and provide confidence context.
        """

        performance = {
            'confidence_level': 'High',
            'reliability_warning': None,
            'recommendation_confidence': 'High'
        }

        if 'accuracy' in self.results:
            accuracy = self.results['accuracy']

            if accuracy < 0.6:
                performance['confidence_level'] = 'Low'
                performance['reliability_warning'] = (
                    f"Model accuracy ({accuracy:.2%}) is below "
                    "recommended threshold (60%)."
                )
                performance['recommendation_confidence'] = 'Low'

            elif accuracy < 0.8:
                performance['confidence_level'] = 'Medium'
                performance['reliability_warning'] = (
                    f"Model accuracy ({accuracy:.2%}) suggests "
                    "moderate reliability."
                )
                performance['recommendation_confidence'] = 'Medium'

        if 'r2' in self.results:
            r2 = self.results['r2']

            if r2 < 0:
                performance['confidence_level'] = 'Very Low'
                performance['reliability_warning'] = (
                    f"Model performs worse than baseline "
                    f"(R² = {r2:.3f})."
                )
                performance['recommendation_confidence'] = 'Very Low'

            elif r2 < 0.3:
                performance['confidence_level'] = 'Low'
                performance['reliability_warning'] = (
                    f"Model explains only {r2:.1%} of variance."
                )
                performance['recommendation_confidence'] = 'Low'

        return performance

    # =========================================================
    # UTILS
    # =========================================================
    def _get_customer_id(self, row: pd.Series):
        for col in [
            'CustomerID',
            'Customer_ID',
            'customer_id',
            'customerid'
        ]:
            if col in row:
                return row[col]

        return 'Unknown'

    def _clean_feature_name(self, feature: str) -> str:
        return (
            str(feature)
            .replace('num__', '')
            .replace('cat__', '')
            .replace('remainder__', '')
        )

    def _top_feature_names(self, n: int = 5) -> list:
        """
        Extract top feature names from feature_importances.
        """

        feature_importances = self.results.get('feature_importances')

        if (
            feature_importances is not None
            and isinstance(feature_importances, pd.DataFrame)
            and not feature_importances.empty
            and 'feature' in feature_importances.columns
        ):
            return [
                self._clean_feature_name(f)
                for f in feature_importances['feature'].head(n).tolist()
            ]

        if 'test_data' in self.results:
            return list(self.results['test_data'].columns[:n])

        return []

    def _build_contributing_factors(
        self,
        row: pd.Series,
        feature_names: list
    ) -> str:
        """
        Build readable contributing factors from row values.
        """

        factors = []

        for feature in feature_names:
            candidates = [
                feature,
                f"num__{feature}",
                f"cat__{feature}",
                f"remainder__{feature}"
            ]

            value_found = False

            for candidate in candidates:
                if candidate in row:
                    value = row[candidate]

                    if isinstance(value, (int, float, np.number)):
                        factors.append(
                            f"{feature}={value:.3f}"
                        )
                    else:
                        factors.append(
                            f"{feature}={value}"
                        )

                    value_found = True
                    break

            if not value_found:
                continue

        return ", ".join(factors)

    def _engagement_status(self, row: pd.Series) -> str:
        """
        Basic marketing engagement interpretation.
        """

        ctr = row.get('ClickThroughRate', None)
        conv_rate = row.get('ConversionRate', None)
        visits = row.get('WebsiteVisits', None)
        pages = row.get('PagesPerVisit', None)
        time_on_site = row.get('TimeOnSite', None)

        notes = []

        if ctr is not None:
            if ctr < 0.05:
                notes.append("low CTR")
            elif ctr > 0.20:
                notes.append("strong CTR")

        if conv_rate is not None:
            if conv_rate < 0.05:
                notes.append("low conversion rate")
            elif conv_rate > 0.15:
                notes.append("strong conversion rate")

        if visits is not None and visits <= 2:
            notes.append("low website visits")

        if pages is not None and pages < 2:
            notes.append("low page depth")

        if time_on_site is not None and time_on_site < 2:
            notes.append("low time on site")

        if not notes:
            return "general campaign performance signals"

        return ", ".join(notes)

    def _marketing_action_from_prediction(
        self,
        row: pd.Series,
        prediction
    ) -> Dict[str, str]:
        """
        Create marketing recommendation based on prediction and campaign signals.
        """

        try:
            pred_num = float(prediction)
        except Exception:
            pred_num = None

        ctr = row.get('ClickThroughRate', None)
        conversion_rate = row.get('ConversionRate', None)
        ad_spend = row.get('AdSpend', None)
        email_opens = row.get('EmailOpens', None)
        email_clicks = row.get('EmailClicks', None)
        visits = row.get('WebsiteVisits', None)
        previous_purchases = row.get('PreviousPurchases', None)
        loyalty_points = row.get('LoyaltyPoints', None)

        channel = row.get('CampaignChannel', 'Unknown')
        campaign_type = row.get('CampaignType', 'Unknown')

        # Predicted non-conversion / weak outcome
        if pred_num == 0:
            if ctr is not None and ctr < 0.05:
                return {
                    "priority_level": "Critical",
                    "action": (
                        "Revise ad creative, audience targeting, and campaign message "
                        "because predicted conversion is low and CTR is weak."
                    ),
                    "decision_area": "Creative & Targeting Optimization",
                    "estimated_impact": "High",
                    "timeframe": "Immediate"
                }

            if (
                email_opens is not None
                and email_clicks is not None
                and email_opens > 5
                and email_clicks <= 2
            ):
                return {
                    "priority_level": "High",
                    "action": (
                        "Improve email CTA, landing page relevance, and offer clarity "
                        "because users open emails but do not click enough."
                    ),
                    "decision_area": "Email Marketing Optimization",
                    "estimated_impact": "Medium-High",
                    "timeframe": "Within 3-7 days"
                }

            if visits is not None and visits > 20:
                return {
                    "priority_level": "High",
                    "action": (
                        "Launch retargeting campaign and optimize landing page flow "
                        "because traffic exists but conversion is predicted low."
                    ),
                    "decision_area": "Retargeting & Landing Page Optimization",
                    "estimated_impact": "Medium-High",
                    "timeframe": "Within 1 week"
                }

            return {
                "priority_level": "High",
                "action": (
                    "Reallocate budget or test alternative channel/campaign type "
                    "because this audience is predicted not to convert."
                ),
                "decision_area": "Budget Reallocation",
                "estimated_impact": "Medium",
                "timeframe": "Within 1 week"
            }

        # Predicted conversion / positive outcome
        if pred_num == 1:
            if (
                conversion_rate is not None
                and conversion_rate > 0.15
                and ad_spend is not None
            ):
                return {
                    "priority_level": "Growth",
                    "action": (
                        "Scale budget gradually for this high-performing campaign "
                        "segment while monitoring cost efficiency."
                    ),
                    "decision_area": "Budget Scaling",
                    "estimated_impact": "High",
                    "timeframe": "Next campaign cycle"
                }

            if (
                previous_purchases is not None
                and previous_purchases >= 5
            ) or (
                loyalty_points is not None
                and loyalty_points >= 3000
            ):
                return {
                    "priority_level": "Growth",
                    "action": (
                        "Prioritize retention, loyalty, and upsell campaigns "
                        "because the customer segment shows strong purchase history."
                    ),
                    "decision_area": "Retention & Upsell Strategy",
                    "estimated_impact": "Medium-High",
                    "timeframe": "Next campaign cycle"
                }

            return {
                "priority_level": "Opportunity",
                "action": (
                    "Maintain campaign exposure and use this segment as a benchmark "
                    "for similar audiences."
                ),
                "decision_area": "Campaign Continuation",
                "estimated_impact": "Medium",
                "timeframe": "Ongoing"
            }

        # Regression-style prediction
        if pred_num is not None:
            if pred_num >= 0.15:
                return {
                    "priority_level": "Growth",
                    "action": (
                        "Increase investment in this segment because predicted "
                        "performance is above average."
                    ),
                    "decision_area": "Budget Scaling",
                    "estimated_impact": "High",
                    "timeframe": "Next campaign cycle"
                }

            if pred_num < 0.05:
                return {
                    "priority_level": "Critical",
                    "action": (
                        "Pause or redesign campaign strategy because predicted "
                        "performance is weak."
                    ),
                    "decision_area": "Campaign Redesign",
                    "estimated_impact": "High",
                    "timeframe": "Immediate"
                }

        return {
            "priority_level": "Advisory",
            "action": (
                f"Review campaign channel '{channel}' and campaign type "
                f"'{campaign_type}' for optimization opportunities."
            ),
            "decision_area": "Marketing Review",
            "estimated_impact": "Medium",
            "timeframe": "Within 2 weeks"
        }

    # =========================================================
    # RECOMMENDATION GENERATION
    # =========================================================
    def generate_recommendations(self) -> pd.DataFrame:
        """
        Generate digital marketing and advertising recommendations.
        """

        logging.info(
            "Generating digital marketing decision recommendations..."
        )

        model_performance = self._assess_model_performance()

        logging.info(
            f"Model performance assessment: {model_performance}"
        )

        train_predictions = self.results.get('train_predictions')

        is_regression = (
            train_predictions is not None
            and np.issubdtype(
                np.asarray(train_predictions).dtype,
                np.number
            )
            and self.results.get('r2') is not None
        )

        # =====================================================
        # REGRESSION RECOMMENDATIONS
        # =====================================================
        if is_regression:
            results_df = self.results['test_data'].copy()
            results_df['Predicted_Value'] = self.results[
                'test_predictions'
            ]

            train_arr = np.asarray(train_predictions)

            train_mean = train_arr.mean()
            train_std = train_arr.std()

            high_threshold = train_mean + train_std
            low_threshold = train_mean - train_std

            recommendations = []

            for _, row in results_df.iterrows():
                predicted_value = row['Predicted_Value']

                if predicted_value >= high_threshold:
                    priority = "Growth"
                    action = (
                        "Increase investment or prioritize this campaign "
                        "segment because predicted performance is high."
                    )
                    decision_area = "Budget Scaling"
                    estimated_impact = "High"
                    timeframe = "Next campaign cycle"

                elif predicted_value <= low_threshold:
                    priority = "Critical"
                    action = (
                        "Review, redesign, or reduce budget for this campaign "
                        "segment because predicted performance is low."
                    )
                    decision_area = "Campaign Redesign"
                    estimated_impact = "High"
                    timeframe = "Immediate"

                else:
                    priority = "Advisory"
                    action = (
                        "Monitor campaign performance and continue controlled testing."
                    )
                    decision_area = "Performance Monitoring"
                    estimated_impact = "Medium"
                    timeframe = "Ongoing"

                recommendations.append({
                    'CustomerID': self._get_customer_id(row),
                    'Predicted_Value': predicted_value,
                    'Priority_Level': priority,
                    'Decision_Area': decision_area,
                    'Recommended_Action': action,
                    'Estimated_Impact': estimated_impact,
                    'Timeframe': timeframe,
                    'CampaignChannel': row.get('CampaignChannel', 'Unknown'),
                    'CampaignType': row.get('CampaignType', 'Unknown'),
                    'Engagement_Status': self._engagement_status(row),
                    'Model_Confidence': model_performance[
                        'recommendation_confidence'
                    ]
                })

            recommendations_df = pd.DataFrame(recommendations)

            if model_performance['reliability_warning']:
                recommendations_df['Model_Warning'] = model_performance[
                    'reliability_warning'
                ]

            return recommendations_df

        # =====================================================
        # ANOMALY DETECTION RECOMMENDATIONS
        # =====================================================
        if 'anomaly_labels' in self.results:
            results_df = self.results['results_df'].copy()

            anomalous = results_df[
                results_df['Is_Anomaly']
            ]

            if anomalous.empty:
                logging.info(
                    "No marketing anomalies detected."
                )

                return pd.DataFrame()

            recommendations = []

            for _, row in anomalous.head(30).iterrows():
                recommendations.append({
                    'CustomerID': row.get('CustomerID', 'Unknown'),
                    'Priority_Level': 'Critical',
                    'Priority_Score': abs(
                        row.get('Anomaly_Score', 0)
                    ),
                    'Decision_Area': 'Anomaly Investigation',
                    'Recommended_Action': (
                        "Investigate unusual marketing/customer behavior "
                        "and verify campaign tracking, audience quality, "
                        "or abnormal engagement pattern."
                    ),
                    'Estimated_Impact': 'High',
                    'Timeframe': 'Immediate',
                    'Contributing_Factors': (
                        "Anomalous pattern detected across marketing features."
                    )
                })

            return pd.DataFrame(recommendations)

        # =====================================================
        # CLASSIFICATION RECOMMENDATIONS
        # =====================================================
        results_df = self.results['test_data'].copy()

        predictions = pd.Series(
            self.results['test_predictions'],
            index=results_df.index,
            name="Predicted_Conversion"
        )

        results_df['Predicted_Conversion'] = predictions

        top_features = self._top_feature_names(5)

        recommendations = []

        for _, row in results_df.iterrows():
            prediction = row['Predicted_Conversion']

            action_plan = self._marketing_action_from_prediction(
                row,
                prediction
            )

            contributing = self._build_contributing_factors(
                row,
                top_features
            )

            engagement_status = self._engagement_status(row)

            reason = (
                f"Model predicted conversion outcome = {prediction}. "
                f"Campaign signals indicate {engagement_status}."
            )

            if top_features:
                reason += (
                    " Key model drivers include: "
                    + ", ".join(top_features)
                    + "."
                )

            recommendations.append({
                'CustomerID': self._get_customer_id(row),
                'Predicted_Conversion': prediction,
                'Priority_Level': action_plan['priority_level'],
                'Decision_Area': action_plan['decision_area'],
                'Recommended_Action': action_plan['action'],
                'Estimated_Impact': action_plan['estimated_impact'],
                'Timeframe': action_plan['timeframe'],
                'CampaignChannel': row.get('CampaignChannel', 'Unknown'),
                'CampaignType': row.get('CampaignType', 'Unknown'),
                'Engagement_Status': engagement_status,
                'Contributing_Factors': (
                    contributing
                    if contributing
                    else "Top feature values unavailable."
                ),
                'Reason_for_Action': reason,
                'Model_Confidence': model_performance[
                    'recommendation_confidence'
                ]
            })

        recommendations_df = pd.DataFrame(recommendations)

        priority_order = {
            'Critical': 0,
            'High': 1,
            'Growth': 2,
            'Opportunity': 3,
            'Advisory': 4
        }

        if not recommendations_df.empty:
            recommendations_df['_priority_rank'] = recommendations_df[
                'Priority_Level'
            ].map(priority_order).fillna(9)

            recommendations_df = recommendations_df.sort_values(
                ['_priority_rank']
            ).drop(columns=['_priority_rank'])

            recommendations_df = recommendations_df.head(30)

        if model_performance['reliability_warning']:
            recommendations_df['Model_Warning'] = model_performance[
                'reliability_warning'
            ]

        if recommendations_df.empty:
            logging.info(
                "No digital marketing recommendations generated."
            )

        return recommendations_df

    # =========================================================
    # SUMMARY REPORT
    # =========================================================
    def generate_summary_report(
        self,
        recommendations_df: pd.DataFrame
    ) -> str:
        """
        Generate a human-readable marketing recommendation summary.
        """

        if recommendations_df.empty:
            return (
                "✅ No urgent digital marketing actions required. "
                "Campaign performance appears stable based on model output."
            )

        report_lines = [
            "🎯 DIGITAL MARKETING DECISION RECOMMENDATION SUMMARY",
            "=" * 60
        ]

        if 'Priority_Level' in recommendations_df.columns:
            priority_counts = recommendations_df[
                'Priority_Level'
            ].value_counts()

            report_lines.append("\n📊 Priority Distribution:")

            for priority, count in priority_counts.items():
                report_lines.append(
                    f"  • {priority}: {count} recommendation(s)"
                )

        if 'Decision_Area' in recommendations_df.columns:
            decision_counts = recommendations_df[
                'Decision_Area'
            ].value_counts()

            report_lines.append("\n📌 Decision Area Distribution:")

            for area, count in decision_counts.items():
                report_lines.append(
                    f"  • {area}: {count} recommendation(s)"
                )

        if 'CampaignChannel' in recommendations_df.columns:
            channel_counts = recommendations_df[
                'CampaignChannel'
            ].value_counts()

            report_lines.append("\n📣 Campaign Channel Focus:")

            for channel, count in channel_counts.head(5).items():
                report_lines.append(
                    f"  • {channel}: {count} recommendation(s)"
                )

        top_recs = recommendations_df.head(5)

        report_lines.append("\n🚀 TOP RECOMMENDED ACTIONS:")

        for _, rec in top_recs.iterrows():
            customer = rec.get('CustomerID', 'Unknown')
            action = rec.get('Recommended_Action', 'N/A')
            area = rec.get('Decision_Area', 'N/A')

            report_lines.append(
                f"  • Customer {customer} | {area}: {action}"
            )

        if (
            'Model_Warning' in recommendations_df.columns
            and not recommendations_df['Model_Warning'].isna().all()
        ):
            warnings = recommendations_df[
                'Model_Warning'
            ].dropna().unique()

            if len(warnings) > 0:
                report_lines.append("\n⚠️ MODEL RELIABILITY WARNING:")

                for warning in warnings:
                    report_lines.append(
                        f"  • {warning}"
                    )

        return "\n".join(report_lines)


if __name__ == '__main__':
    logging.info(
        "--- Running Digital Marketing Optimization Agent "
        "in Standalone Mode ---"
    )

    sample_analysis_results = {
        'test_data': pd.DataFrame({
            'CustomerID': [8000, 8001, 8002, 8003],
            'Age': [56, 69, 46, 32],
            'Income': [136912, 41760, 88456, 44085],
            'CampaignChannel': [
                'Social Media',
                'Email',
                'PPC',
                'PPC'
            ],
            'CampaignType': [
                'Awareness',
                'Retention',
                'Awareness',
                'Conversion'
            ],
            'AdSpend': [6497.87, 3898.66, 1546.42, 539.52],
            'ClickThroughRate': [0.043, 0.155, 0.277, 0.137],
            'ConversionRate': [0.088, 0.182, 0.076, 0.088],
            'WebsiteVisits': [0, 42, 2, 47],
            'EmailOpens': [6, 2, 11, 2],
            'EmailClicks': [9, 7, 2, 2],
            'PreviousPurchases': [4, 2, 8, 0],
            'LoyaltyPoints': [688, 3459, 2337, 2463]
        }),
        'test_predictions': [0, 1, 1, 0],
        'feature_importances': pd.DataFrame({
            'feature': [
                'num__ClickThroughRate',
                'num__ConversionRate',
                'num__AdSpend'
            ],
            'importance': [0.4, 0.35, 0.25]
        })
    }

    optimization_agent = OptimizationAgent(
        sample_analysis_results
    )

    recs = optimization_agent.generate_recommendations()

    logging.info("\n" + recs.to_string())

    logging.info(
        "\n" + optimization_agent.generate_summary_report(recs)
    )

    logging.info("--- End of Standalone Run ---")