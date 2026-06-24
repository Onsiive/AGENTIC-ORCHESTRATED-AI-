import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype, is_categorical_dtype, is_bool_dtype
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

class IntelligentFeatureAnalyzer:
    """
    Advanced feature analysis and selection system that provides intelligent
    insights about feature relationships, importance, and selection strategies.
    """
    
    def __init__(self, target_column: str, problem_type: str):
        """
        Initialize the intelligent feature analyzer.
        
        Args:
            target_column (str): Name of the target column
            problem_type (str): 'classification', 'regression', or 'anomaly_detection'
        """
        self.target_column = target_column
        self.problem_type = problem_type
        self.feature_insights = {}
        self.correlation_matrix = None
        self.feature_importance_scores = {}
        
    def analyze_features(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform comprehensive feature analysis.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            Dict[str, Any]: Comprehensive feature analysis results
        """
        logging.info("Starting intelligent feature analysis...")
        
        # 1. Correlation Analysis
        correlation_insights = self._analyze_correlations(X, y)
        
        # 2. Feature Importance Analysis
        importance_insights = self._analyze_feature_importance(X, y)
        
        # 3. Mutual Information Analysis
        mutual_info_insights = self._analyze_mutual_information(X, y)
        
        # 4. Feature Redundancy Detection
        redundancy_insights = self._detect_feature_redundancy(X)
        
        # 5. Generate Recommendations
        recommendations = self._generate_feature_recommendations(
            correlation_insights, importance_insights, 
            mutual_info_insights, redundancy_insights
        )
        
        self.feature_insights = {
            'correlations': correlation_insights,
            'importance': importance_insights,
            'mutual_information': mutual_info_insights,
            'redundancy': redundancy_insights,
            'recommendations': recommendations,
            'summary': self._generate_summary(recommendations)
        }
        
        logging.info("Feature analysis completed successfully")
        return self.feature_insights
    
    def _analyze_correlations(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze feature correlations and target relationships."""
        logging.info("Analyzing feature correlations...")
        
        # Calculate correlation matrix
        numeric_features = X.select_dtypes(include=[np.number]).columns
        correlation_matrix = X[numeric_features].corr()
        self.correlation_matrix = correlation_matrix
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                if corr_val > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        # Calculate target correlations
        target_correlations = {}
        if self.problem_type in ['classification', 'regression']:
            for feature in numeric_features:
                if self.problem_type == 'classification':
                    # For classification, use point-biserial correlation
                    corr_val = X[feature].corr(y.astype('category').cat.codes)
                else:
                    corr_val = X[feature].corr(y)
                target_correlations[feature] = corr_val
        
        return {
            'correlation_matrix': correlation_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'target_correlations': target_correlations,
            'max_correlation': correlation_matrix.abs().max().max(),
            'avg_correlation': correlation_matrix.abs().mean().mean()
        }
    
    def _analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze feature importance using multiple methods."""
        logging.info("Analyzing feature importance...")
        
        # Prepare data
        X_encoded = self._encode_categorical_features(X)
        
        if self.problem_type == 'classification':
            # Random Forest importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_encoded, y)
            rf_importance = dict(zip(X_encoded.columns, rf.feature_importances_))
            
            # F-score importance
            f_scores, _ = f_classif(X_encoded, y)
            f_importance = dict(zip(X_encoded.columns, f_scores))
            
        elif self.problem_type == 'regression':
            # Random Forest importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_encoded, y)
            rf_importance = dict(zip(X_encoded.columns, rf.feature_importances_))
            
            # F-score importance
            f_scores, _ = f_regression(X_encoded, y)
            f_importance = dict(zip(X_encoded.columns, f_scores))
        
        else:  # anomaly_detection
            # For anomaly detection, use variance and range analysis
            rf_importance = {}
            f_importance = {}
            for feature in X_encoded.columns:
                rf_importance[feature] = X_encoded[feature].var()
                f_importance[feature] = X_encoded[feature].max() - X_encoded[feature].min()
        
        # Normalize importance scores
        rf_importance = self._normalize_scores(rf_importance)
        f_importance = self._normalize_scores(f_importance)
        
        # Combine scores
        combined_importance = {}
        for feature in X_encoded.columns:
            combined_importance[feature] = (
                rf_importance.get(feature, 0) * 0.6 + 
                f_importance.get(feature, 0) * 0.4
            )
        
        self.feature_importance_scores = combined_importance
        
        return {
            'random_forest_importance': rf_importance,
            'f_score_importance': f_importance,
            'combined_importance': combined_importance,
            'top_features': sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _analyze_mutual_information(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Analyze mutual information between features and target.

        For operational stability, this method uses a fast proxy when the
        feature matrix is wider than a small exploratory case. The previous
        direct use of sklearn mutual_info_* could become very slow/hang on
        mixed digital-marketing feature tables in some sklearn/Python builds.
        The proxy still preserves the intent of the module: ranking features
        by target association for auditable feature evaluation.
        """
        logging.info("Analyzing mutual information...")

        X_encoded = self._encode_categorical_features(X)

        try:
            if X_encoded.shape[1] <= 5 and X_encoded.shape[0] >= 20:
                if self.problem_type == 'classification':
                    mi_scores = mutual_info_classif(
                        X_encoded,
                        y,
                        random_state=42,
                        n_neighbors=3
                    )
                elif self.problem_type == 'regression':
                    mi_scores = mutual_info_regression(
                        X_encoded,
                        y,
                        random_state=42,
                        n_neighbors=3
                    )
                else:
                    mi_scores = X_encoded.var().values
            else:
                raise RuntimeError("Using fast target-association proxy for MI")

            mi_dict = dict(zip(X_encoded.columns, mi_scores))

        except Exception as exc:
            logging.info(
                "Using fast mutual-information proxy based on target association: "
                f"{exc}"
            )

            if self.problem_type == 'classification':
                try:
                    scores, _ = f_classif(X_encoded, y)
                    mi_dict = dict(zip(X_encoded.columns, scores))
                except Exception:
                    y_encoded = LabelEncoder().fit_transform(y.astype(str))
                    mi_dict = {
                        col: abs(pd.Series(X_encoded[col]).corr(pd.Series(y_encoded)))
                        for col in X_encoded.columns
                    }
            elif self.problem_type == 'regression':
                try:
                    scores, _ = f_regression(X_encoded, y)
                    mi_dict = dict(zip(X_encoded.columns, scores))
                except Exception:
                    y_numeric = pd.to_numeric(y, errors='coerce').fillna(0)
                    mi_dict = {
                        col: abs(pd.Series(X_encoded[col]).corr(y_numeric))
                        for col in X_encoded.columns
                    }
            else:
                mi_dict = X_encoded.var().to_dict()

        mi_dict = self._normalize_scores(mi_dict)

        return {
            'mutual_information_scores': mi_dict,
            'top_mi_features': sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _detect_feature_redundancy(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Detect redundant features based on correlation and variance."""
        logging.info("Detecting feature redundancy...")
        
        numeric_features = X.select_dtypes(include=[np.number]).columns
        redundant_features = []
        low_variance_features = []
        
        # Check for low variance features
        for feature in numeric_features:
            variance = X[feature].var()
            unique_count = X[feature].nunique(dropna=True)

            # Use a near-zero threshold only. Marketing rate features such as
            # CTR and ConversionRate naturally have variance below 0.01, but
            # they are semantically important and must not be dropped.
            if unique_count <= 1 or variance < 1e-8:
                low_variance_features.append({
                    'feature': feature,
                    'variance': variance,
                    'reason': 'Near-zero variance or constant feature'
                })
        
        # Check for highly correlated features
        if self.correlation_matrix is not None:
            for i in range(len(self.correlation_matrix.columns)):
                for j in range(i+1, len(self.correlation_matrix.columns)):
                    corr_val = abs(self.correlation_matrix.iloc[i, j])
                    if corr_val > 0.95:  # Very high correlation
                        feature1 = self.correlation_matrix.columns[i]
                        feature2 = self.correlation_matrix.columns[j]
                        
                        # Keep the one with higher variance
                        var1 = X[feature1].var()
                        var2 = X[feature2].var()
                        
                        if var1 > var2:
                            redundant_features.append({
                                'redundant_feature': feature2,
                                'keep_feature': feature1,
                                'correlation': corr_val,
                                'reason': f'Highly correlated with {feature1} (r={corr_val:.3f})'
                            })
                        else:
                            redundant_features.append({
                                'redundant_feature': feature1,
                                'keep_feature': feature2,
                                'correlation': corr_val,
                                'reason': f'Highly correlated with {feature2} (r={corr_val:.3f})'
                            })
        
        return {
            'redundant_features': redundant_features,
            'low_variance_features': low_variance_features,
            'total_redundant': len(redundant_features) + len(low_variance_features)
        }
    
    def _generate_feature_recommendations(self, corr_insights, importance_insights, 
                                        mi_insights, redundancy_insights) -> Dict[str, Any]:
        """Generate intelligent feature selection recommendations."""
        logging.info("Generating feature recommendations...")
        
        recommendations = {
            'features_to_remove': [],
            'features_to_keep': [],
            'features_to_investigate': [],
            'feature_engineering_suggestions': [],
            'modeling_recommendations': []
        }
        
        # Remove redundant features
        for redundant in redundancy_insights['redundant_features']:
            recommendations['features_to_remove'].append({
                'feature': redundant['redundant_feature'],
                'reason': redundant['reason'],
                'priority': 'High'
            })
        
        for low_var in redundancy_insights['low_variance_features']:
            recommendations['features_to_remove'].append({
                'feature': low_var['feature'],
                'reason': low_var['reason'],
                'priority': 'High'
            })
        
        # Keep top important features
        top_features = importance_insights['top_features'][:10]  # Top 10 features
        for feature, score in top_features:
            recommendations['features_to_keep'].append({
                'feature': feature,
                'importance_score': score,
                'reason': f'High importance score ({score:.3f})',
                'priority': 'High'
            })
        
        # Investigate features with mixed signals
        all_features = set(importance_insights['combined_importance'].keys())
        keep_features = set([f['feature'] for f in recommendations['features_to_keep']])
        remove_features = set([f['feature'] for f in recommendations['features_to_remove']])
        
        investigate_features = all_features - keep_features - remove_features
        for feature in investigate_features:
            recommendations['features_to_investigate'].append({
                'feature': feature,
                'importance': importance_insights['combined_importance'].get(feature, 0),
                'reason': 'Mixed signals - needs manual review'
            })
        
        # Feature engineering suggestions
        if corr_insights['high_correlation_pairs']:
            recommendations['feature_engineering_suggestions'].append({
                'suggestion': 'Consider creating interaction terms',
                'details': f"Found {len(corr_insights['high_correlation_pairs'])} highly correlated feature pairs",
                'priority': 'Medium'
            })
        
        if corr_insights['max_correlation'] > 0.9:
            recommendations['feature_engineering_suggestions'].append({
                'suggestion': 'High multicollinearity detected',
                'details': f"Maximum correlation: {corr_insights['max_correlation']:.3f}",
                'priority': 'High'
            })
        
        # Modeling recommendations
        if len(recommendations['features_to_remove']) > 0:
            recommendations['modeling_recommendations'].append({
                'recommendation': 'Remove redundant features before modeling',
                'details': f"Remove {len(recommendations['features_to_remove'])} redundant features",
                'priority': 'High'
            })
        
        if len(recommendations['features_to_keep']) < 5:
            recommendations['modeling_recommendations'].append({
                'recommendation': 'Consider feature engineering',
                'details': f"Only {len(recommendations['features_to_keep'])} high-importance features identified",
                'priority': 'Medium'
            })
        
        return recommendations
    
    def _generate_summary(self, recommendations: Dict[str, Any]) -> str:
        """Generate a human-readable summary of feature analysis."""
        summary_lines = [
            "🧠 INTELLIGENT FEATURE ANALYSIS SUMMARY",
            "=" * 50
        ]
        
        # Feature counts
        total_remove = len(recommendations['features_to_remove'])
        total_keep = len(recommendations['features_to_keep'])
        total_investigate = len(recommendations['features_to_investigate'])
        
        summary_lines.append(f"\n📊 Feature Recommendations:")
        summary_lines.append(f"  • Remove: {total_remove} redundant/low-variance features")
        summary_lines.append(f"  • Keep: {total_keep} high-importance features")
        summary_lines.append(f"  • Investigate: {total_investigate} features need review")
        
        # Top features
        if recommendations['features_to_keep']:
            summary_lines.append(f"\n⭐ Top Features to Keep:")
            for i, feature in enumerate(recommendations['features_to_keep'][:5], 1):
                summary_lines.append(f"  {i}. {feature['feature']} (score: {feature['importance_score']:.3f})")
        
        # Redundant features
        if recommendations['features_to_remove']:
            summary_lines.append(f"\n🗑️ Features to Remove:")
            for feature in recommendations['features_to_remove'][:5]:
                summary_lines.append(f"  • {feature['feature']}: {feature['reason']}")
        
        # Engineering suggestions
        if recommendations['feature_engineering_suggestions']:
            summary_lines.append(f"\n🔧 Feature Engineering Suggestions:")
            for suggestion in recommendations['feature_engineering_suggestions']:
                summary_lines.append(f"  • {suggestion['suggestion']}: {suggestion['details']}")
        
        return "\n".join(summary_lines)
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode every non-numeric feature into numeric values for scikit-learn.

        The digital marketing dataset contains categorical campaign fields such
        as Gender, CampaignChannel, CampaignType, AdvertisingPlatform, and
        AdvertisingTool. Some pandas versions load these columns as object,
        while others may load them as string/category/bool dtypes. The previous
        implementation only checked dtype == 'object', so StringDtype columns
        could leak into RandomForest/f_classif and trigger errors such as:
        ValueError: could not convert string to float: 'Social Media'.
        """
        X_encoded = X.copy()

        for column in X_encoded.columns:
            series = X_encoded[column]

            if is_numeric_dtype(series):
                X_encoded[column] = pd.to_numeric(series, errors='coerce')
                X_encoded[column] = X_encoded[column].replace([np.inf, -np.inf], np.nan)
                if X_encoded[column].isna().any():
                    X_encoded[column] = X_encoded[column].fillna(X_encoded[column].median())
                continue

            if (
                is_object_dtype(series)
                or is_string_dtype(series)
                or is_categorical_dtype(series)
                or is_bool_dtype(series)
            ):
                le = LabelEncoder()
                safe_values = series.astype('string').fillna('__MISSING__')
                X_encoded[column] = le.fit_transform(safe_values)
                continue

            # Last-resort coercion for unusual dtypes such as dates.
            coerced = pd.to_numeric(series, errors='coerce')
            if coerced.isna().all():
                le = LabelEncoder()
                safe_values = series.astype('string').fillna('__MISSING__')
                X_encoded[column] = le.fit_transform(safe_values)
            else:
                X_encoded[column] = coerced.fillna(coerced.median())

        return X_encoded
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to 0-1 range with NaN/inf protection."""
        if not scores:
            return {}

        clean_scores = {}
        for key, value in scores.items():
            try:
                numeric_value = float(value)
            except Exception:
                numeric_value = 0.0

            if np.isnan(numeric_value) or np.isinf(numeric_value):
                numeric_value = 0.0

            clean_scores[key] = numeric_value

        max_score = max(clean_scores.values())
        min_score = min(clean_scores.values())

        if max_score == min_score:
            return {k: 1.0 for k in clean_scores.keys()}

        return {
            k: (v - min_score) / (max_score - min_score)
            for k, v in clean_scores.items()
        }
    
    def get_feature_selection_mask(self, X: pd.DataFrame) -> List[bool]:
        """Get boolean mask for feature selection based on analysis."""
        if not self.feature_insights:
            return [True] * len(X.columns)
        
        recommendations = self.feature_insights['recommendations']
        features_to_remove = [f['feature'] for f in recommendations['features_to_remove']]
        
        return [col not in features_to_remove for col in X.columns]

if __name__ == '__main__':
    # Example usage
    logging.info("--- Testing Intelligent Feature Analyzer ---")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'redundant_feature': np.random.normal(0, 1, n_samples),  # Will be highly correlated with feature1
        'low_variance': np.random.normal(0, 0.001, n_samples),  # Low variance
        'target': np.random.randint(0, 3, n_samples)  # Classification target
    }
    
    # Make redundant_feature highly correlated with feature1
    data['redundant_feature'] = data['feature1'] + np.random.normal(0, 0.1, n_samples)
    
    df = pd.DataFrame(data)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Test the analyzer
    analyzer = IntelligentFeatureAnalyzer('target', 'classification')
    results = analyzer.analyze_features(X, y)
    
    print("\n" + results['summary'])
    print("\nFeature selection mask:", analyzer.get_feature_selection_mask(X))
