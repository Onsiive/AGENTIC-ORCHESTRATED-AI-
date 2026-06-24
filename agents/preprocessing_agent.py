import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import sparse
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tool_decider import get_tool_decider, create_data_summary, ToolDecider
from utils.intelligent_feature_analysis import IntelligentFeatureAnalyzer

# Configure logging for the module
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s'
)


class PreprocessingAgent:
    """
    PreprocessingAgent is responsible for cleaning and preparing the dataset.

    This version is adapted for digital marketing and advertising analytics.
    It prepares campaign/customer features for conversion prediction,
    campaign performance modeling, and recommendation generation.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tool_decider: Optional[ToolDecider] = None,
        target_column: Optional[str] = None,
        problem_type: Optional[str] = None,
        protected_columns: Optional[list] = None
    ):
        """
        Initialize the PreprocessingAgent.

        Args:
            data:
                Raw data to be preprocessed.

            tool_decider:
                Tool decider for preprocessing strategy selection.

            target_column:
                Target column name. Example: Conversion.

            problem_type:
                classification or regression.

            protected_columns:
                Columns explicitly selected by planner that should not be
                automatically removed unless they are unsafe identifiers.
        """

        logging.info("Initializing Preprocessing Agent...")

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        self.data = data.copy()
        self.tool_decider = tool_decider or get_tool_decider("rule_based")
        self.target_column = target_column
        self.problem_type = problem_type
        self.protected_columns = protected_columns or []

        self.feature_analyzer = None
        self.last_feature_insights = None
        self.feature_insights = None
        self.identifier_columns = []

        logging.info(
            f"Preprocessing Agent initialized with data of shape: "
            f"{self.data.shape}"
        )

    # =========================================================
    # FEATURE TYPE DETECTION
    # =========================================================
    def get_feature_types(self):
        """
        Identifies numerical and categorical features in the dataset.
        """

        numerical_features = self.data.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        categorical_features = self.data.select_dtypes(
            include=['object', 'string', 'category', 'bool']
        ).columns.tolist()

        # Target should never be treated as feature
        if self.target_column in numerical_features:
            numerical_features.remove(self.target_column)

        if self.target_column in categorical_features:
            categorical_features.remove(self.target_column)

        if self.problem_type == 'anomaly_detection':
            kept_identifiers = []
            protected_set = set(self.protected_columns)

            for feature in list(numerical_features):
                upper_name = feature.upper()

                if (
                    (
                        'ID' in upper_name
                        or upper_name.endswith('_ID')
                        or upper_name == 'ID'
                    )
                    and feature in protected_set
                ):
                    kept_identifiers.append(feature)
                    numerical_features.remove(feature)

            if kept_identifiers:
                logging.info(
                    f"Protected identifier columns for anomaly detection: "
                    f"{kept_identifiers}"
                )

                passthrough_df = self.data[kept_identifiers].copy()
                passthrough_df.columns = [
                    f"identifier__{col}"
                    for col in kept_identifiers
                ]

                self.data = pd.concat(
                    [self.data, passthrough_df],
                    axis=1
                )

                self.identifier_columns = [
                    f"identifier__{col}"
                    for col in kept_identifiers
                ]

            else:
                self.identifier_columns = []

        else:
            self.identifier_columns = []

        logging.info(
            f"Identified Numerical Features: {numerical_features}"
        )

        logging.info(
            f"Identified Categorical Features: {categorical_features}"
        )

        return numerical_features, categorical_features

    # =========================================================
    # INTELLIGENT FEATURE ANALYSIS
    # =========================================================
    def perform_intelligent_feature_analysis(self) -> Dict[str, Any]:
        """
        Perform intelligent feature analysis if target column and problem type
        are available.
        """

        if (
            not self.target_column
            or not self.problem_type
            or self.target_column not in self.data.columns
        ):
            logging.info(
                "Skipping intelligent feature analysis - "
                "target column or problem type not available"
            )

            return {}

        logging.info(
            "Performing intelligent feature analysis for "
            "digital marketing dataset..."
        )

        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        self.feature_analyzer = IntelligentFeatureAnalyzer(
            self.target_column,
            self.problem_type
        )

        self.feature_insights = self.feature_analyzer.analyze_features(
            X,
            y
        )

        logging.info("Intelligent feature analysis completed")

        if (
            isinstance(self.feature_insights, dict)
            and 'summary' in self.feature_insights
        ):
            logging.info(
                f"Analysis summary:\n"
                f"{self.feature_insights['summary']}"
            )

        return self.feature_insights

    # =========================================================
    # MARKETING-SPECIFIC COLUMN FILTER
    # =========================================================
    def get_marketing_columns_to_drop(self) -> list:
        """
        Drop columns that are unsafe, non-informative, or not useful
        for digital marketing predictive modeling.
        """

        cols_to_drop = []

        for col in self.data.columns:
            col_lower = col.lower()
            col_upper = col.upper()

            # Always remove target from feature preprocessing
            if self.target_column and col == self.target_column:
                cols_to_drop.append(col)
                logging.info(
                    f"Removing target column '{col}' "
                    "to prevent data leakage"
                )
                continue

            # Drop direct identifiers
            if (
                col_lower == "customerid"
                or col_lower == "customer_id"
                or col_upper == "CUSTOMERID"
                or col_upper == "CUSTOMER_ID"
            ):
                cols_to_drop.append(col)
                logging.info(
                    f"Dropping customer identifier column '{col}'"
                )
                continue

            # Generic ID handling
            if (
                'ID' in col_upper
                and col not in self.protected_columns
                and not col.startswith("identifier__")
            ):
                cols_to_drop.append(col)
                logging.info(
                    f"Dropping identifier column '{col}'"
                )
                continue

            # Drop columns with only one unique value
            nunique = self.data[col].nunique(dropna=True)

            if nunique <= 1:
                cols_to_drop.append(col)
                logging.info(
                    f"Dropping static/non-informative column "
                    f"'{col}' with {nunique} unique value"
                )
                continue

            # Drop fully empty columns
            if self.data[col].isna().all():
                cols_to_drop.append(col)
                logging.info(
                    f"Dropping fully empty column '{col}'"
                )
                continue

        return cols_to_drop

    # =========================================================
    # PREPROCESSING PIPELINE
    # =========================================================
    def create_preprocessing_pipeline(
        self,
        numerical_features,
        categorical_features
    ):
        """
        Creates a scikit-learn preprocessing pipeline.

        Numeric:
            imputation + scaling

        Categorical:
            imputation + one-hot encoding
        """

        logging.info("Creating preprocessing pipeline...")

        data_summary = create_data_summary(self.data)

        available_tools = [
            "imputation",
            "scaling",
            "encoding",
            "normalization"
        ]

        decision = self.tool_decider.decide_preprocessing_strategy(
            data_summary,
            available_tools
        )

        logging.info(
            f"ToolDecider chose preprocessing strategy: {decision}"
        )

        # ----------------------------
        # Numerical transformer
        # ----------------------------
        numerical_steps = []

        if "imputation" in decision.get("tools", []):
            if data_summary.get("missing_percentage", 0) > 20:
                numerical_steps.append(
                    ('imputer', KNNImputer(n_neighbors=3))
                )

                logging.info(
                    "Using KNN imputation for high missing percentage"
                )

            else:
                numerical_steps.append(
                    ('imputer', SimpleImputer(strategy='median'))
                )

                logging.info(
                    "Using median imputation for numeric features"
                )

        if "scaling" in decision.get("tools", []):
            if data_summary.get("memory_usage_mb", 0) > 100:
                numerical_steps.append(
                    ('scaler', RobustScaler())
                )

                logging.info(
                    "Using RobustScaler for large or noisy dataset"
                )

            else:
                numerical_steps.append(
                    ('scaler', StandardScaler())
                )

                logging.info(
                    "Using StandardScaler for numeric features"
                )

        elif "normalization" in decision.get("tools", []):
            numerical_steps.append(
                ('scaler', MinMaxScaler())
            )

            logging.info(
                "Using MinMaxScaler for numeric features"
            )

        # ----------------------------
        # Categorical transformer
        # ----------------------------
        categorical_steps = []

        if "imputation" in decision.get("tools", []):
            categorical_steps.append(
                ('imputer', SimpleImputer(strategy='most_frequent'))
            )

        valid_categorical_features = []

        if categorical_features and "encoding" not in decision.get("tools", []):
            decision.setdefault("tools", []).append("encoding")
            logging.info(
                "Encoding forced because categorical campaign features are present."
            )

        if "encoding" in decision.get("tools", []):
            for feature in categorical_features:
                unique_count = self.data[feature].nunique(dropna=True)

                if unique_count > 50:
                    logging.warning(
                        f"High cardinality categorical feature "
                        f"'{feature}' ({unique_count} unique values) "
                        "will be dropped to prevent feature explosion"
                    )

                else:
                    valid_categorical_features.append(feature)

                    logging.info(
                        f"Categorical feature '{feature}' "
                        f"({unique_count} unique values) "
                        "is safe for one-hot encoding"
                    )

            if valid_categorical_features:
                categorical_steps.append(
                    (
                        'onehot',
                        OneHotEncoder(
                            handle_unknown='ignore',
                            max_categories=50
                        )
                    )
                )

            else:
                logging.warning(
                    "No categorical features suitable for encoding"
                )

        numerical_transformer = (
            Pipeline(steps=numerical_steps)
            if numerical_steps
            else None
        )

        categorical_transformer = (
            Pipeline(steps=categorical_steps)
            if categorical_steps
            else None
        )

        transformers = []

        if numerical_transformer and numerical_features:
            transformers.append(
                (
                    'num',
                    numerical_transformer,
                    numerical_features
                )
            )

        if categorical_transformer and valid_categorical_features:
            transformers.append(
                (
                    'cat',
                    categorical_transformer,
                    valid_categorical_features
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )

        logging.info(
            f"Preprocessing pipeline created successfully "
            f"with {len(transformers)} transformer(s)."
        )

        return preprocessor

    # =========================================================
    # MAIN PREPROCESS METHOD
    # =========================================================
    def preprocess(self) -> Optional[pd.DataFrame]:
        """
        Executes the full preprocessing pipeline on the data.
        """

        logging.info(
            "Starting digital marketing data preprocessing..."
        )

        try:
            feature_insights = (
                self.perform_intelligent_feature_analysis()
            )

            self.last_feature_insights = feature_insights

            numerical_features, categorical_features = (
                self.get_feature_types()
            )

            cols_to_drop = self.get_marketing_columns_to_drop()

            # Drop high-cardinality categorical columns
            for feature in categorical_features:
                if feature not in self.protected_columns:
                    unique_count = self.data[feature].nunique(dropna=True)

                    if (
                        unique_count > 50
                        and not feature.startswith("identifier__")
                    ):
                        cols_to_drop.append(feature)

                        logging.warning(
                            f"Dropping high-cardinality feature "
                            f"'{feature}' ({unique_count} unique values)"
                        )

            # Use intelligent feature recommendations if available
            if (
                feature_insights
                and isinstance(feature_insights, dict)
                and 'recommendations' in feature_insights
            ):
                recommendations = feature_insights['recommendations']

                for feature_rec in recommendations.get(
                    'features_to_remove',
                    []
                ):
                    feature_name = feature_rec.get('feature')

                    if (
                        feature_name in self.data.columns
                        and feature_name != self.target_column
                    ):
                        cols_to_drop.append(feature_name)

                        logging.info(
                            f"Intelligently removing feature "
                            f"'{feature_name}': "
                            f"{feature_rec.get('reason', '')}"
                        )

                for suggestion in recommendations.get(
                    'feature_engineering_suggestions',
                    []
                ):
                    logging.info(
                        f"Feature engineering suggestion: "
                        f"{suggestion.get('suggestion', '')} - "
                        f"{suggestion.get('details', '')}"
                    )

            # Remove duplicates while preserving order
            seen = set()
            cols_to_drop_unique = []

            for col in cols_to_drop:
                if col in self.data.columns and col not in seen:
                    cols_to_drop_unique.append(col)
                    seen.add(col)

            if cols_to_drop_unique:
                logging.info(
                    f"Dropping columns before preprocessing: "
                    f"{cols_to_drop_unique}"
                )

                self.data = self.data.drop(
                    columns=cols_to_drop_unique
                )

                numerical_features = [
                    f for f in numerical_features
                    if f not in cols_to_drop_unique
                ]

                categorical_features = [
                    f for f in categorical_features
                    if f not in cols_to_drop_unique
                ]

            passthrough_identifiers = []

            if self.protected_columns:
                for col in self.protected_columns:
                    if (
                        col in categorical_features
                        and 'ID' in col.upper()
                    ):
                        passthrough_identifiers.append(col)

                        logging.info(
                            f"Keeping identifier column '{col}' "
                            "as pass-through"
                        )

            if passthrough_identifiers:
                categorical_features = [
                    f for f in categorical_features
                    if f not in passthrough_identifiers
                ]

            if not numerical_features and not categorical_features:
                logging.error(
                    "No usable features remain after preprocessing filters."
                )

                return None

            pipeline = self.create_preprocessing_pipeline(
                numerical_features,
                categorical_features
            )

            logging.info(
                "Fitting and transforming the data with preprocessing pipeline..."
            )

            processed_data = pipeline.fit_transform(self.data)

            try:
                feature_names = pipeline.get_feature_names_out()
            except Exception:
                feature_names = [
                    f"feature_{i}"
                    for i in range(processed_data.shape[1])
                ]

            if sparse.issparse(processed_data):
                processed_df = pd.DataFrame.sparse.from_spmatrix(
                    processed_data,
                    index=self.data.index,
                    columns=feature_names
                )

            else:
                processed_df = pd.DataFrame(
                    processed_data,
                    columns=feature_names,
                    index=self.data.index
                )

            logging.info(
                f"Data preprocessing complete. "
                f"New data shape: {processed_df.shape}"
            )

            return processed_df

        except Exception as e:
            logging.error(
                f"An error occurred during preprocessing: {e}",
                exc_info=True
            )

            return None


if __name__ == '__main__':
    logging.info(
        "--- Running Preprocessing Agent in Standalone Mode ---"
    )

    sample_data = {
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
            0.0439,
            0.1557,
            0.2774,
            0.1376,
            0.2528
        ],
        'ConversionRate': [
            0.088,
            0.182,
            0.076,
            0.088,
            0.109
        ],
        'Conversion': [1, 1, 1, 1, 1]
    }

    sample_df = pd.DataFrame(sample_data)

    preprocessing_agent = PreprocessingAgent(
        sample_df,
        target_column='Conversion',
        problem_type='classification'
    )

    preprocessed_data = preprocessing_agent.preprocess()

    if preprocessed_data is not None:
        logging.info("--- Preprocessed Data ---")
        logging.info("\n" + preprocessed_data.to_string())
        logging.info("--- End of Standalone Run ---")