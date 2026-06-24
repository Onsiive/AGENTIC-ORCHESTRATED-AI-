"""
ToolDecider - tactical decision module for agent-level tool selection.

This module provides a standardized interface for deciding:
- preprocessing strategy
- model selection
- hyperparameters

This version is adapted for digital marketing and advertising analytics.
"""

import logging
import json
import pandas as pd
import sys
import os
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.local_llm_agent import LocalLLMAgent

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s'
)


class ToolDecider(ABC):
    """
    Abstract base class for tool decision making.
    """

    @abstractmethod
    def decide_preprocessing_strategy(
        self,
        data_summary: Dict[str, Any],
        available_tools: List[str]
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def decide_model_family(
        self,
        task_type: str,
        data_summary: Dict[str, Any],
        available_models: List[str]
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def decide_hyperparameters(
        self,
        model: str,
        task_type: str,
        data_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        pass


class RuleBasedToolDecider(ToolDecider):
    """
    Rule-based tool decider.

    Used as fallback when no LLM is available.
    Rules are adapted for digital marketing datasets.
    """

    def decide_preprocessing_strategy(
        self,
        data_summary: Dict[str, Any],
        available_tools: List[str]
    ) -> Dict[str, Any]:

        missing_pct = data_summary.get('missing_percentage', 0)
        numeric_count = data_summary.get('numeric_columns', 0)
        categorical_count = data_summary.get('categorical_columns', 0)

        tools = []

        if missing_pct > 0:
            tools.append("imputation")

        if numeric_count > 0:
            tools.append("scaling")

        if categorical_count > 0:
            tools.append("encoding")

        if not tools:
            tools = ["passthrough"]

        return {
            "strategy": "digital_marketing_standard",
            "tools": tools,
            "reason": (
                f"Digital marketing preprocessing selected: "
                f"{missing_pct:.1f}% missing values, "
                f"{numeric_count} numeric columns, "
                f"{categorical_count} categorical columns. "
                "Numeric campaign/customer features are scaled and "
                "categorical campaign attributes are encoded."
            )
        }

    def decide_model_family(
        self,
        task_type: str,
        data_summary: Dict[str, Any],
        available_models: List[str]
    ) -> Dict[str, Any]:

        n_samples = data_summary.get('n_samples', 0)
        n_features = data_summary.get('n_features', 0)
        target_name = str(data_summary.get('target_column', '')).lower()
        class_balance = data_summary.get('class_balance', {})

        if task_type == "classification":
            # Digital marketing conversion prediction often benefits from tree models
            # because campaign response behavior can be nonlinear.
            if "GradientBoostingClassifier" in available_models:
                model = "GradientBoostingClassifier"
            elif "RandomForestClassifier" in available_models:
                model = "RandomForestClassifier"
            elif "LogisticRegression" in available_models:
                model = "LogisticRegression"
            else:
                model = available_models[0]

            reason = (
                "Selected model for digital marketing classification "
                "such as conversion prediction. Tree/boosting models are "
                "preferred for nonlinear relationships among ad spend, CTR, "
                "engagement, channel, and customer behavior."
            )

            if class_balance:
                reason += (
                    f" Class balance detected: {class_balance}."
                )

        elif task_type == "regression":
            # Useful for ConversionRate / CTR prediction.
            if "RandomForestRegressor" in available_models:
                model = "RandomForestRegressor"
            elif "GradientBoostingRegressor" in available_models:
                model = "GradientBoostingRegressor"
            elif "Ridge" in available_models and n_features > 50:
                model = "Ridge"
            elif "LinearRegression" in available_models:
                model = "LinearRegression"
            else:
                model = available_models[0]

            reason = (
                "Selected regression model for marketing performance "
                "prediction such as CTR, conversion rate, or campaign score."
            )

        elif task_type == "anomaly_detection":
            if "IsolationForest" in available_models:
                model = "IsolationForest"
            else:
                model = available_models[0]

            reason = (
                "Selected anomaly detection model for identifying unusual "
                "campaign or customer behavior."
            )

        else:
            if "RandomForestClassifier" in available_models:
                model = "RandomForestClassifier"
            else:
                model = available_models[0]

            reason = (
                "Fallback model selected for unspecified task."
            )

        return {
            "model": model,
            "reason": (
                f"{reason} Dataset summary: "
                f"{n_samples} samples, {n_features} features, "
                f"target={target_name or 'unknown'}."
            )
        }

    def decide_hyperparameters(
        self,
        model: str,
        task_type: str,
        data_summary: Dict[str, Any]
    ) -> Dict[str, Any]:

        n_samples = data_summary.get('n_samples', 1000)
        n_features = data_summary.get('n_features', 10)

        if model == "RandomForestClassifier":
            n_estimators = 150 if n_samples >= 5000 else 100

            return {
                "n_estimators": n_estimators,
                "max_depth": None,
                "random_state": 42,
                "class_weight": "balanced",
                "reason": (
                    "RandomForestClassifier tuned for conversion prediction "
                    "with balanced class handling."
                )
            }

        if model == "GradientBoostingClassifier":
            return {
                "n_estimators": 120,
                "learning_rate": 0.05,
                "max_depth": 3,
                "random_state": 42,
                "reason": (
                    "GradientBoostingClassifier selected for nonlinear "
                    "campaign performance patterns."
                )
            }

        if model == "LogisticRegression":
            return {
                "max_iter": 2000,
                "C": 1.0,
                "class_weight": "balanced",
                "random_state": 42,
                "reason": (
                    "LogisticRegression configured as interpretable baseline "
                    "for conversion prediction."
                )
            }

        if model == "RandomForestRegressor":
            n_estimators = 150 if n_samples >= 5000 else 100

            return {
                "n_estimators": n_estimators,
                "max_depth": None,
                "random_state": 42,
                "reason": (
                    "RandomForestRegressor configured for marketing metric "
                    "prediction with nonlinear feature interactions."
                )
            }

        if model == "GradientBoostingRegressor":
            return {
                "n_estimators": 120,
                "learning_rate": 0.05,
                "max_depth": 3,
                "random_state": 42,
                "reason": (
                    "GradientBoostingRegressor configured for campaign "
                    "performance prediction."
                )
            }

        if model == "Ridge":
            return {
                "alpha": 1.0,
                "random_state": 42,
                "reason": (
                    "Ridge regression used for stable linear marketing "
                    "metric prediction."
                )
            }

        if model == "Lasso":
            return {
                "alpha": 0.01,
                "random_state": 42,
                "reason": (
                    "Lasso regression used for feature selection and "
                    "marketing metric prediction."
                )
            }

        if model == "IsolationForest":
            return {
                "contamination": "auto",
                "n_estimators": 200,
                "random_state": 42,
                "reason": (
                    "IsolationForest configured for unusual campaign/customer "
                    "behavior detection."
                )
            }

        return {
            "random_state": 42,
            "reason": (
                "Default hyperparameters selected for the given model."
            )
        }


class LLMToolDecider(ToolDecider):
    """
    LLM-based tool decider using local models.
    """

    def __init__(
        self,
        llm_agent: Optional[LocalLLMAgent] = None,
        fallback_decider: Optional[ToolDecider] = None
    ):
        self.llm_agent = llm_agent
        self.fallback_decider = fallback_decider or RuleBasedToolDecider()

    def _query_llm(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:

        try:
            if self.llm_agent is None:
                return None

            full_prompt = (
                f"{prompt}\n\n"
                f"Context:\n{json.dumps(context, indent=2)}\n\n"
                "Return ONLY valid JSON. "
                "Do not include markdown. "
                "Do not include explanation outside JSON."
            )

            response = self.llm_agent.generate(
                full_prompt,
                max_tokens=512
            )

            parsed = response.get('parsed')

            if parsed:
                return parsed

            raw_text = response.get('raw', '')

            start = raw_text.find('{')
            end = raw_text.rfind('}')

            if start != -1 and end > start:
                json_str = raw_text[start:end + 1]
                return json.loads(json_str)

        except Exception as e:
            logging.warning(
                f"LLM query failed: {e}"
            )

        return None

    def decide_preprocessing_strategy(
        self,
        data_summary: Dict[str, Any],
        available_tools: List[str]
    ) -> Dict[str, Any]:

        prompt = """
You are a data preprocessing expert for digital marketing analytics.

Choose the best preprocessing strategy for campaign/customer data.

Available tools:
imputation, scaling, encoding, normalization, feature_selection, outlier_detection

Use this logic:
- Impute missing numeric/categorical values when needed.
- Scale numeric columns such as Age, Income, AdSpend, CTR, ConversionRate.
- Encode categorical columns such as Gender, CampaignChannel, CampaignType.
- Avoid using CustomerID as a predictive feature.
- Avoid leaking target columns such as Conversion into features.

Respond with JSON:
{"strategy":"strategy_name","tools":["tool1","tool2"],"reason":"short explanation"}
"""

        context = {
            "domain": "digital_marketing_and_advertising",
            "data_summary": data_summary,
            "available_tools": available_tools
        }

        result = self._query_llm(prompt, context)

        if result:
            return result

        logging.info(
            "LLM preprocessing decision failed, using rule-based fallback."
        )

        return self.fallback_decider.decide_preprocessing_strategy(
            data_summary,
            available_tools
        )

    def decide_model_family(
        self,
        task_type: str,
        data_summary: Dict[str, Any],
        available_models: List[str]
    ) -> Dict[str, Any]:

        prompt = f"""
You are a machine learning expert for digital marketing analytics.

For a {task_type} task, choose the best model family.

Marketing use cases:
- Conversion prediction
- Campaign performance prediction
- CTR or ConversionRate prediction
- Channel/campaign effectiveness analysis

Available models:
{', '.join(available_models)}

Consider:
- sample size
- feature count
- categorical campaign features
- nonlinear customer behavior
- interpretability
- computational efficiency

Respond with JSON:
{{"model":"model_name","reason":"short explanation"}}
"""

        context = {
            "domain": "digital_marketing_and_advertising",
            "task_type": task_type,
            "data_summary": data_summary,
            "available_models": available_models
        }

        result = self._query_llm(prompt, context)

        if result:
            model = result.get("model")

            if model in available_models:
                return result

            logging.warning(
                f"LLM selected unsupported model '{model}'. "
                "Using fallback."
            )

        logging.info(
            "LLM model decision failed, using rule-based fallback."
        )

        return self.fallback_decider.decide_model_family(
            task_type,
            data_summary,
            available_models
        )

    def decide_hyperparameters(
        self,
        model: str,
        task_type: str,
        data_summary: Dict[str, Any]
    ) -> Dict[str, Any]:

        prompt = f"""
You are a hyperparameter tuning expert for digital marketing analytics.

Suggest appropriate hyperparameters for:
model={model}
task={task_type}

Consider:
- campaign dataset size
- feature count
- classification imbalance
- conversion prediction reliability
- computational constraints

Respond with JSON only.

Common parameters:
- RandomForestClassifier: n_estimators, max_depth, class_weight, random_state
- GradientBoostingClassifier: n_estimators, learning_rate, max_depth, random_state
- LogisticRegression: max_iter, C, class_weight, random_state
- RandomForestRegressor: n_estimators, max_depth, random_state
- IsolationForest: contamination, n_estimators, random_state
"""

        context = {
            "domain": "digital_marketing_and_advertising",
            "model": model,
            "task_type": task_type,
            "data_summary": data_summary
        }

        result = self._query_llm(prompt, context)

        if result:
            return result

        logging.info(
            "LLM hyperparameter decision failed, using rule-based fallback."
        )

        return self.fallback_decider.decide_hyperparameters(
            model,
            task_type,
            data_summary
        )


def get_tool_decider(
    decider_type: str = "rule_based",
    llm_agent: Optional[LocalLLMAgent] = None,
    **kwargs
) -> ToolDecider:

    if decider_type.lower() == "rule_based":
        return RuleBasedToolDecider()

    if decider_type.lower() == "llm":
        return LLMToolDecider(
            llm_agent=llm_agent
        )

    if decider_type.lower() == "hybrid":
        return LLMToolDecider(
            llm_agent=llm_agent,
            fallback_decider=RuleBasedToolDecider()
        )

    raise ValueError(
        f"Unknown tool decider type: {decider_type}"
    )


def create_data_summary(
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Create a summary of dataset characteristics for tool decision making.
    """

    numeric_cols = df.select_dtypes(
        include=['number']
    ).columns.tolist()

    categorical_cols = df.select_dtypes(
        include=['object', 'string', 'category', 'bool']
    ).columns.tolist()

    summary = {
        "n_samples": int(len(df)),
        "n_features": int(len(df.columns)),
        "numeric_columns": int(len(numeric_cols)),
        "categorical_columns": int(len(categorical_cols)),
        "missing_percentage": float(df.isnull().mean().mean() * 100),
        "has_missing_values": bool(df.isnull().any().any()),
        "memory_usage_mb": float(
            df.memory_usage(deep=True).sum() / 1024 / 1024
        ),
        "dtypes": df.dtypes.apply(
            lambda x: str(x.name)
        ).to_dict(),
        "numeric_cols": [
            str(col) for col in numeric_cols
        ],
        "categorical_cols": [
            str(col) for col in categorical_cols
        ],
        "possible_identifier_cols": [
            str(col)
            for col in df.columns
            if (
                "id" in str(col).lower()
                or str(col).lower().endswith("_id")
            )
        ],
        "possible_marketing_cols": [
            str(col)
            for col in df.columns
            if any(
                keyword in str(col).lower()
                for keyword in [
                    "campaign",
                    "channel",
                    "ad",
                    "click",
                    "conversion",
                    "visit",
                    "email",
                    "social",
                    "loyalty",
                    "income",
                    "age"
                ]
            )
        ]
    }

    if "Conversion" in df.columns:
        summary["target_column"] = "Conversion"

        try:
            summary["class_balance"] = (
                df["Conversion"]
                .value_counts(normalize=True)
                .round(4)
                .to_dict()
            )

        except Exception:
            summary["class_balance"] = {}

    elif "ConversionRate" in df.columns:
        summary["target_column"] = "ConversionRate"

    elif "ClickThroughRate" in df.columns:
        summary["target_column"] = "ClickThroughRate"

    return summary


if __name__ == "__main__":
    test_data = pd.DataFrame({
        'CustomerID': [8000, 8001, 8002, 8003, 8004],
        'Age': [56, 69, 46, 32, 60],
        'Gender': ['Female', 'Male', 'Female', 'Female', 'Female'],
        'Income': [136912, 41760, 88456, 44085, 83964],
        'CampaignChannel': [
            'Social Media',
            'Email',
            'PPC',
            'PPC',
            'PPC'
        ],
        'CampaignType': [
            'Awareness',
            'Retention',
            'Awareness',
            'Conversion',
            'Conversion'
        ],
        'AdSpend': [
            6497.87,
            3898.66,
            1546.42,
            539.52,
            1678.04
        ],
        'ClickThroughRate': [
            0.043,
            0.155,
            0.277,
            0.137,
            0.252
        ],
        'ConversionRate': [
            0.088,
            0.182,
            0.076,
            0.088,
            0.109
        ],
        'Conversion': [1, 1, 1, 0, 1]
    })

    summary = create_data_summary(test_data)

    print("Data Summary:")
    print(json.dumps(summary, indent=2))

    print("\n=== Testing Rule-based Decider ===")

    rule_decider = RuleBasedToolDecider()

    preprocessing = rule_decider.decide_preprocessing_strategy(
        summary,
        ["imputation", "scaling", "encoding"]
    )

    print(
        "Preprocessing decision:",
        json.dumps(preprocessing, indent=2)
    )

    model = rule_decider.decide_model_family(
        "classification",
        summary,
        [
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "LogisticRegression"
        ]
    )

    print(
        "Model decision:",
        json.dumps(model, indent=2)
    )

    hyperparams = rule_decider.decide_hyperparameters(
        "RandomForestClassifier",
        "classification",
        summary
    )

    print(
        "Hyperparameters:",
        json.dumps(hyperparams, indent=2)
    )