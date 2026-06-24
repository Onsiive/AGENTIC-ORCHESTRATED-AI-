import logging
import pandas as pd
from typing import Optional, Dict, Any
import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    IsolationForest,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso
)
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tool_decider import get_tool_decider, create_data_summary, ToolDecider

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s'
)


class DynamicAnalysisAgent:
    """
    DynamicAnalysisAgent performs model selection and analysis.

    This version is adapted for digital marketing and advertising analytics:
    - conversion prediction
    - campaign performance modeling
    - feature importance analysis
    - marketing decision support
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        task: str = "classification",
        params: Dict[str, Any] = None,
        tool_decider: Optional[ToolDecider] = None
    ):
        self.data = data
        self.target_column = target_column
        self.task = task
        self.params = params or {}
        self.tool_decider = tool_decider or get_tool_decider("rule_based")

        self.model = None
        self.model_name = None
        self.results = {}

        self.tried_models = []
        self.best_performance = -float('inf')
        self.best_model = None
        self.best_results = None

        logging.info(
            f"DynamicAnalysisAgent initialized for task: "
            f"{task} with params: {self.params}"
        )

    # =========================================================
    # MODEL SELECTION
    # =========================================================
    def choose_tool(self) -> str:
        """
        Select analysis model based on task type.
        """

        if self.task == "anomaly_detection":
            self.model_name = "IsolationForest"

            logging.info(
                f"Anomaly detection task, selected tool: "
                f"{self.model_name}"
            )

            return self.model_name

        data_summary = create_data_summary(self.data)

        if self.task == "classification":
            available_models = [
                "RandomForestClassifier",
                "GradientBoostingClassifier",
                "LogisticRegression",
                "SVC"
            ]

            default_model = "RandomForestClassifier"

        elif self.task == "regression":
            available_models = [
                "RandomForestRegressor",
                "GradientBoostingRegressor",
                "LinearRegression",
                "Ridge",
                "Lasso",
                "SVR"
            ]

            default_model = "RandomForestRegressor"

        else:
            available_models = [
                "RandomForestClassifier",
                "LogisticRegression"
            ]

            default_model = "RandomForestClassifier"

        decision = self.tool_decider.decide_model_family(
            self.task,
            data_summary,
            available_models
        )

        self.model_name = decision.get(
            "model",
            default_model
        )

        if self.model_name not in available_models:
            logging.warning(
                f"ToolDecider selected unsupported model "
                f"'{self.model_name}', using default: {default_model}"
            )

            self.model_name = default_model

        logging.info(
            f"ToolDecider selected tool: {self.model_name}, "
            f"reason: {decision.get('reason', 'N/A')}"
        )

        return self.model_name

    # =========================================================
    # MAIN RUN
    # =========================================================
    def run(self, force_retry: bool = False) -> Optional[Dict[str, Any]]:
        """
        Execute selected model and return analysis results.
        """

        if self.task == "anomaly_detection":
            return self._run_anomaly_detection()

        if self.target_column is None:
            logging.error(
                "Target column required for supervised learning tasks."
            )

            return None

        if self.target_column not in self.data.columns:
            logging.error(
                f"Target column '{self.target_column}' "
                f"not found in data."
            )

            return None

        if force_retry:
            return self._try_multiple_models()

        tool = self.choose_tool()

        if tool == "LogisticRegression":
            return self._run_logistic_regression()

        elif tool == "SVC":
            return self._run_svc()

        elif tool == "RandomForestClassifier":
            return self._run_random_forest()

        elif tool == "GradientBoostingClassifier":
            return self._run_gradient_boosting_classifier()

        elif tool == "RandomForestRegressor":
            return self._run_random_forest_regressor()

        elif tool == "GradientBoostingRegressor":
            return self._run_gradient_boosting_regressor()

        elif tool == "LinearRegression":
            return self._run_linear_regression()

        elif tool == "Ridge":
            return self._run_ridge()

        elif tool == "Lasso":
            return self._run_lasso()

        elif tool == "SVR":
            return self._run_svr()

        else:
            if self.task == "regression":
                return self._run_random_forest_regressor()

            return self._run_random_forest()

    # =========================================================
    # DATA HELPERS
    # =========================================================
    def _prepare_supervised_data(self):
        """
        Prepare X and y for supervised learning.

        Drops target and identifier-like columns.
        """

        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        id_columns = [
            col for col in X.columns
            if (
                'ID' in col.upper()
                or 'CUSTOMERID' in col.upper()
                or 'CUSTOMER_ID' in col.upper()
            )
        ]

        if id_columns:
            logging.info(
                f"Dropping ID columns from model training: {id_columns}"
            )

            X = X.drop(columns=id_columns)

        # Convert sparse dataframe columns if needed
        for col in X.columns:
            if pd.api.types.is_sparse(X[col]):
                X[col] = X[col].sparse.to_dense()

        return X, y

    def _classification_train_test_split(self, X, y):
        """
        Train-test split with stratification when possible.
        """

        try:
            return train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

        except Exception:
            logging.warning(
                "Stratified split failed, using regular split."
            )

            return train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

    def _safe_auc(self, model, X_test, y_test):
        """
        Compute ROC-AUC safely for binary classification.
        """

        try:
            if len(np.unique(y_test)) != 2:
                return None

            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X_test)[:, 1]
                return roc_auc_score(y_test, scores)

            if hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
                return roc_auc_score(y_test, scores)

        except Exception as e:
            logging.warning(
                f"ROC-AUC calculation failed: {e}"
            )

        return None

    def _classification_result(
        self,
        model_name,
        model,
        X_train,
        X_test,
        y_train,
        y_test
    ) -> Dict[str, Any]:

        preds = model.predict(X_test)
        train_preds = model.predict(X_train)

        acc = accuracy_score(y_test, preds)

        precision = precision_score(
            y_test,
            preds,
            average='weighted',
            zero_division=0
        )

        recall = recall_score(
            y_test,
            preds,
            average='weighted',
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            preds,
            average='weighted',
            zero_division=0
        )

        report = classification_report(
            y_test,
            preds,
            output_dict=True,
            zero_division=0
        )

        conf_matrix = confusion_matrix(
            y_test,
            preds
        )

        roc_auc = self._safe_auc(
            model,
            X_test,
            y_test
        )

        feature_importances = None

        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_

        elif hasattr(model, "coef_"):
            coef = model.coef_

            if len(coef.shape) > 1:
                feature_importances = np.mean(
                    np.abs(coef),
                    axis=0
                )

            else:
                feature_importances = np.abs(coef)

        logging.info(
            f"{model_name} - "
            f"Accuracy: {acc:.4f}, "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"F1: {f1:.4f}"
        )

        if roc_auc is not None:
            logging.info(
                f"{model_name} - ROC-AUC: {roc_auc:.4f}"
            )

        return {
            "model": model_name,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "classification_report": report,
            "confusion_matrix": conf_matrix,
            "predictions": preds,
            "train_predictions": train_preds,
            "X_test": X_test,
            "y_test": y_test,
            "fitted_model": model,
            "feature_importances": feature_importances,
            "feature_names": X_test.columns.tolist()
        }

    def _regression_result(
        self,
        model_name,
        model,
        X_train,
        X_test,
        y_train,
        y_test
    ) -> Dict[str, Any]:

        preds = model.predict(X_test)
        train_preds = model.predict(X_train)

        mse = mean_squared_error(y_test, preds)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, preds)

        feature_importances = None

        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_

        elif hasattr(model, "coef_"):
            coef = model.coef_
            feature_importances = np.abs(coef)

        logging.info(
            f"{model_name} - "
            f"MSE: {mse:.4f}, "
            f"RMSE: {rmse:.4f}, "
            f"R²: {r2:.4f}"
        )

        return {
            "model": model_name,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "predictions": preds,
            "train_predictions": train_preds,
            "X_test": X_test,
            "y_test": y_test,
            "fitted_model": model,
            "feature_importances": feature_importances,
            "feature_names": X_test.columns.tolist()
        }

    # =========================================================
    # ADAPTIVE MODEL COMPARISON
    # =========================================================
    def _try_multiple_models(self) -> Optional[Dict[str, Any]]:
        """
        Try multiple models and return the best performing one.
        """

        logging.info(
            "ADAPTIVE INTELLIGENCE: Trying multiple models "
            "for better marketing analysis performance..."
        )

        if self.task == "classification":
            model_candidates = [
                ("RandomForestClassifier", self._run_random_forest),
                ("GradientBoostingClassifier", self._run_gradient_boosting_classifier),
                ("LogisticRegression", self._run_logistic_regression),
                ("SVC", self._run_svc)
            ]

        elif self.task == "regression":
            model_candidates = [
                ("RandomForestRegressor", self._run_random_forest_regressor),
                ("GradientBoostingRegressor", self._run_gradient_boosting_regressor),
                ("LinearRegression", self._run_linear_regression),
                ("Ridge", self._run_ridge),
                ("Lasso", self._run_lasso),
                ("SVR", self._run_svr)
            ]

        else:
            return None

        best_performance = -float('inf')
        best_model_name = None
        best_results = None

        for model_name, model_func in model_candidates:
            if model_name in self.tried_models:
                logging.info(
                    f"Skipping {model_name} because it was already tried."
                )
                continue

            try:
                logging.info(
                    f"Trying {model_name}..."
                )

                results = model_func()

                if results:
                    if self.task == "classification":
                        performance = results.get(
                            "f1_score",
                            results.get("accuracy", 0)
                        )

                    else:
                        performance = results.get(
                            "r2",
                            -float('inf')
                        )

                    logging.info(
                        f"{model_name} performance: "
                        f"{performance:.4f}"
                    )

                    if performance > best_performance:
                        best_performance = performance
                        best_model_name = model_name
                        best_results = results
                        self.best_model = model_name
                        self.best_results = results
                        self.best_performance = performance

                self.tried_models.append(model_name)

            except Exception as e:
                logging.warning(
                    f"{model_name} failed: {str(e)}"
                )

                self.tried_models.append(model_name)
                continue

        if best_results and best_performance > -float('inf'):
            logging.info(
                f"Best model: {best_model_name} "
                f"(performance: {best_performance:.4f})"
            )

            return best_results

        logging.error(
            "All models failed or produced invalid results."
        )

        return None

    # =========================================================
    # FULL BENCHMARK — semua model dijalankan & dibandingkan
    # =========================================================
    def run_full_benchmark(self) -> Optional[Dict[str, Any]]:
        """
        Jalankan SEMUA model klasifikasi/regresi yang relevan,
        kumpulkan metrik masing-masing, lalu kembalikan:
          - 'best_results'     : dict hasil model terbaik (format sama dgn run())
          - 'benchmark_table'  : list of dict berisi metrik semua model
          - 'best_model_name'  : nama model terbaik

        Digunakan untuk keperluan penelitian agar setiap model
        dapat dipertanggungjawabkan secara kuantitatif.
        """
        import sys
        import time

        if self.task not in ("classification", "regression"):
            logging.warning(
                "run_full_benchmark: task harus classification atau regression."
            )
            return self.run()

        if self.task == "classification":
            candidates = [
                ("RandomForestClassifier",     self._run_random_forest),
                ("GradientBoostingClassifier", self._run_gradient_boosting_classifier),
                ("LogisticRegression",         self._run_logistic_regression),
                ("SVC",                        self._run_svc),
            ]
            perf_key = "f1_score"
        else:
            candidates = [
                ("RandomForestRegressor",      self._run_random_forest_regressor),
                ("GradientBoostingRegressor",  self._run_gradient_boosting_regressor),
                ("LinearRegression",           self._run_linear_regression),
                ("Ridge",                      self._run_ridge),
                ("Lasso",                      self._run_lasso),
            ]
            perf_key = "r2"

        total = len(candidates)
        benchmark_table = []
        best_performance = -float('inf')
        best_model_name  = None
        best_results     = None

        # ── Progress bar header ──────────────────────────────
        _sep = "─" * 62
        print()
        print(_sep)
        print("  🔬  BENCHMARK SEMUA MODEL — sedang memproses...")
        print(_sep)

        for idx, (model_name, model_func) in enumerate(candidates, start=1):
            pct = int((idx - 1) / total * 20)
            bar = "█" * pct + "░" * (20 - pct)
            # overwrite baris yang sama
            sys.stdout.write(
                f"\r  [{bar}] ({idx}/{total}) Menjalankan {model_name:<35}"
            )
            sys.stdout.flush()

            t0 = time.time()
            try:
                results = model_func()
                elapsed = time.time() - t0

                if results is None:
                    raise ValueError("model_func returned None")

                row = {"model": model_name, "elapsed_s": round(elapsed, 2)}

                if self.task == "classification":
                    row["accuracy"]  = round(results.get("accuracy",  0), 4)
                    row["precision"] = round(results.get("precision", 0), 4)
                    row["recall"]    = round(results.get("recall",    0), 4)
                    row["f1_score"]  = round(results.get("f1_score",  0), 4)
                    roc = results.get("roc_auc")
                    row["roc_auc"]   = round(roc, 4) if roc is not None else "N/A"
                    performance      = row["f1_score"]
                else:
                    row["r2"]   = round(results.get("r2",   0), 4)
                    row["mse"]  = round(results.get("mse",  0), 4)
                    row["rmse"] = round(results.get("rmse", 0), 4)
                    performance = row["r2"]

                row["status"] = "✅ OK"
                benchmark_table.append(row)

                if performance > best_performance:
                    best_performance = performance
                    best_model_name  = model_name
                    best_results     = results

            except Exception as exc:
                elapsed = time.time() - t0
                logging.warning(f"Benchmark: {model_name} gagal — {exc}")
                row = {
                    "model":     model_name,
                    "elapsed_s": round(elapsed, 2),
                    "status":    f"❌ Error: {str(exc)[:40]}"
                }
                benchmark_table.append(row)

        # progress selesai
        bar_done = "█" * 20
        sys.stdout.write(
            f"\r  [{bar_done}] ({total}/{total}) Selesai.{' ' * 40}\n"
        )
        sys.stdout.flush()
        print(_sep)
        print()

        # ── Cetak tabel perbandingan ─────────────────────────
        _print_benchmark_table(benchmark_table, self.task, best_model_name)

        if best_results is None:
            logging.error("run_full_benchmark: semua model gagal.")
            return None

        # Sisipkan benchmark_table ke dalam hasil terbaik
        # agar bisa diteruskan ke laporan
        best_results = dict(best_results)
        best_results["benchmark_table"]  = benchmark_table
        best_results["best_model_name"]  = best_model_name
        best_results["best_performance"] = best_performance

        return best_results




    def _run_random_forest(self) -> Dict[str, Any]:
        X, y = self._prepare_supervised_data()

        X_train, X_test, y_train, y_test = (
            self._classification_train_test_split(X, y)
        )

        data_summary = create_data_summary(self.data)

        hyperparams = self.tool_decider.decide_hyperparameters(
            "RandomForestClassifier",
            "classification",
            data_summary
        )

        n_estimators = hyperparams.get('n_estimators', 150)
        max_depth = hyperparams.get('max_depth', None)
        random_state = hyperparams.get('random_state', 42)

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        return self._classification_result(
            "RandomForestClassifier",
            self.model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    def _run_gradient_boosting_classifier(self) -> Dict[str, Any]:
        X, y = self._prepare_supervised_data()

        X_train, X_test, y_train, y_test = (
            self._classification_train_test_split(X, y)
        )

        self.model = GradientBoostingClassifier(
            random_state=42
        )

        self.model.fit(X_train, y_train)

        return self._classification_result(
            "GradientBoostingClassifier",
            self.model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    def _run_logistic_regression(self) -> Dict[str, Any]:
        X, y = self._prepare_supervised_data()

        X_train, X_test, y_train, y_test = (
            self._classification_train_test_split(X, y)
        )

        self.model = LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            solver='lbfgs'
        )

        self.model.fit(X_train, y_train)

        return self._classification_result(
            "LogisticRegression",
            self.model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    def _run_svc(self) -> Dict[str, Any]:
        X, y = self._prepare_supervised_data()

        X_train, X_test, y_train, y_test = (
            self._classification_train_test_split(X, y)
        )

        self.model = SVC(
            probability=True,
            class_weight='balanced'
        )

        self.model.fit(X_train, y_train)

        return self._classification_result(
            "SVC",
            self.model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    # =========================================================
    # REGRESSION MODELS
    # =========================================================
    def _run_random_forest_regressor(self) -> Dict[str, Any]:
        X, y = self._prepare_supervised_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        data_summary = create_data_summary(self.data)

        hyperparams = self.tool_decider.decide_hyperparameters(
            "RandomForestRegressor",
            "regression",
            data_summary
        )

        n_estimators = hyperparams.get('n_estimators', 150)
        max_depth = hyperparams.get('max_depth', None)
        random_state = hyperparams.get('random_state', 42)

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        return self._regression_result(
            "RandomForestRegressor",
            self.model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    def _run_gradient_boosting_regressor(self) -> Dict[str, Any]:
        X, y = self._prepare_supervised_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        self.model = GradientBoostingRegressor(
            random_state=42
        )

        self.model.fit(X_train, y_train)

        return self._regression_result(
            "GradientBoostingRegressor",
            self.model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    def _run_linear_regression(self) -> Dict[str, Any]:
        X, y = self._prepare_supervised_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        return self._regression_result(
            "LinearRegression",
            self.model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    def _run_ridge(self) -> Dict[str, Any]:
        X, y = self._prepare_supervised_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        alpha = self.params.get('alpha', 1.0)

        self.model = Ridge(alpha=alpha)
        self.model.fit(X_train, y_train)

        return self._regression_result(
            "Ridge",
            self.model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    def _run_lasso(self) -> Dict[str, Any]:
        X, y = self._prepare_supervised_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        alpha = self.params.get('alpha', 1.0)

        self.model = Lasso(
            alpha=alpha,
            max_iter=5000
        )

        self.model.fit(X_train, y_train)

        return self._regression_result(
            "Lasso",
            self.model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    def _run_svr(self) -> Dict[str, Any]:
        X, y = self._prepare_supervised_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        kernel = self.params.get('kernel', 'rbf')
        C = self.params.get('C', 1.0)

        self.model = SVR(
            kernel=kernel,
            C=C
        )

        self.model.fit(X_train, y_train)

        return self._regression_result(
            "SVR",
            self.model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    # =========================================================
    # ANOMALY DETECTION - KEPT FOR BACKWARD COMPATIBILITY
    # =========================================================
    def _run_anomaly_detection(self) -> Dict[str, Any]:
        X = self.data.copy()

        id_columns = [
            col for col in X.columns
            if (
                'ID' in col.upper()
                or 'CUSTOMERID' in col.upper()
                or 'CUSTOMER_ID' in col.upper()
            )
        ]

        if id_columns:
            logging.info(
                f"Dropping ID columns from anomaly detection: {id_columns}"
            )

            X = X.drop(columns=id_columns)

        for col in X.columns:
            if pd.api.types.is_sparse(X[col]):
                X[col] = X[col].sparse.to_dense()

        cont = (
            self.params.get('contamination')
            if getattr(self, 'params', None)
            else None
        )

        n_est = (
            self.params.get('n_estimators')
            if getattr(self, 'params', None)
            else None
        )

        if cont is None:
            cont = 0.1

        if n_est is None:
            n_est = 200

        if isinstance(cont, str) and cont == 'auto':
            temp_if = IsolationForest(
                random_state=42,
                n_estimators=64,
                max_samples='auto'
            )

            temp_if.fit(X)

            scores = temp_if.score_samples(X)
            thresh = np.percentile(scores, 5)

            cont_est = float(
                (scores < thresh).mean()
            )

            cont = min(
                max(cont_est, 0.001),
                0.2
            )

            logging.info(
                f"Estimated contamination = {cont:.4f}"
            )

        self.model = IsolationForest(
            random_state=42,
            contamination=cont,
            n_estimators=int(n_est),
            max_samples='auto'
        )

        preds = self.model.fit_predict(X)
        anomaly_scores = self.model.score_samples(X)

        n_anomalies = sum(preds == -1)

        customer_col = None

        for col in self.data.columns:
            if (
                'CUSTOMERID' in col.upper()
                or 'CUSTOMER_ID' in col.upper()
            ):
                customer_col = col
                break

        results_df = pd.DataFrame({
            'CustomerID': (
                self.data[customer_col]
                if customer_col
                else pd.Series(range(len(X)))
            ),
            'Anomaly_Score': anomaly_scores,
            'Is_Anomaly': preds == -1
        })

        for col in X.columns:
            if X[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                results_df[f'{col}_Value'] = X[col]

                std = X[col].std()

                if std and std != 0:
                    results_df[f'{col}_zscore'] = (
                        (X[col] - X[col].mean()) / std
                    )

        results_df = results_df.sort_values(
            'Anomaly_Score'
        )

        logging.info(
            f"IsolationForest detected {n_anomalies} "
            f"anomalies out of {len(preds)} samples "
            f"({(n_anomalies / len(preds) * 100):.1f}%)."
        )

        return {
            "model": "IsolationForest",
            "anomaly_labels": preds,
            "feature_names": X.columns.tolist(),
            "n_anomalies": n_anomalies,
            "anomaly_scores": anomaly_scores,
            "results_df": results_df,
            "X": X
        }




# =========================================================
# HELPER — cetak tabel benchmark ke console
# =========================================================
def _print_benchmark_table(
    benchmark_table: list,
    task: str,
    best_model_name: str
) -> None:
    """Cetak tabel perbandingan semua model secara ringkas."""

    sep_thick = "═" * 76
    sep_thin  = "─" * 76

    print()
    print(sep_thick)
    print("  📊  TABEL PERBANDINGAN MODEL (BENCHMARK KUANTITATIF)")
    print(f"      Task: {task.upper()}")
    print(sep_thick)

    if task == "classification":
        header = (
            f"  {'Model':<34} {'Acc':>6} {'Prec':>6} "
            f"{'Rec':>6} {'F1':>6} {'AUC':>6}  {'Waktu':>6}"
        )
        print(header)
        print(sep_thin)
        for row in benchmark_table:
            if row.get("status", "").startswith("❌"):
                flag = "  ❌"
            elif row["model"] == best_model_name:
                flag = "  ⭐"
            else:
                flag = "    "

            acc  = f"{row.get('accuracy',  'err'):>6}" if isinstance(row.get('accuracy'),  float) else f"{'err':>6}"
            prec = f"{row.get('precision', 'err'):>6}" if isinstance(row.get('precision'), float) else f"{'err':>6}"
            rec  = f"{row.get('recall',    'err'):>6}" if isinstance(row.get('recall'),    float) else f"{'err':>6}"
            f1   = f"{row.get('f1_score',  'err'):>6}" if isinstance(row.get('f1_score'),  float) else f"{'err':>6}"
            auc  = f"{row.get('roc_auc',   'N/A'):>6}" if isinstance(row.get('roc_auc'),   float) else f"{'N/A':>6}"
            t    = f"{row.get('elapsed_s', 0):>5.1f}s"

            label = row['model']
            if row['model'] == best_model_name:
                label += " ← TERBAIK"
            print(f"{flag} {label:<34} {acc} {prec} {rec} {f1} {auc}  {t}")

    else:  # regression
        header = (
            f"  {'Model':<38} {'R²':>8} {'MSE':>10} {'RMSE':>8}  {'Waktu':>6}"
        )
        print(header)
        print(sep_thin)
        for row in benchmark_table:
            flag = "  ⭐" if row["model"] == best_model_name else "    "
            r2   = f"{row.get('r2',   'err'):>8}" if isinstance(row.get('r2'),   float) else f"{'err':>8}"
            mse  = f"{row.get('mse',  'err'):>10}" if isinstance(row.get('mse'),  float) else f"{'err':>10}"
            rmse = f"{row.get('rmse', 'err'):>8}" if isinstance(row.get('rmse',  float)) else f"{'err':>8}"
            t    = f"{row.get('elapsed_s', 0):>5.1f}s"
            label = row['model']
            if row['model'] == best_model_name:
                label += " ← TERBAIK"
            print(f"{flag} {label:<38} {r2} {mse} {rmse}  {t}")

    print(sep_thin)
    if best_model_name:
        print(f"  ⭐ Model terbaik: {best_model_name}")
    print(sep_thick)
    print()




if __name__ == "__main__":
    logging.info(
        "--- Running DynamicAnalysisAgent in Standalone Mode ---"
    )

    df = pd.DataFrame({
        'Age': [22, 35, 41, 29, 53, 48, 31, 39, 45, 27],
        'Income': [30000, 60000, 80000, 45000, 90000, 75000, 52000, 68000, 72000, 40000],
        'AdSpend': [100, 200, 250, 130, 300, 270, 180, 220, 260, 120],
        'ClickThroughRate': [0.04, 0.10, 0.13, 0.06, 0.15, 0.12, 0.08, 0.11, 0.14, 0.05],
        'Conversion': [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    })

    agent = DynamicAnalysisAgent(
        df,
        target_column='Conversion',
        task='classification'
    )

    results = agent.run()

    logging.info(f"Results: {results}")