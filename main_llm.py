import logging
from agents.llm_planner_agent import LLMPlannerAgent
import sys
import argparse
import os
import pandas as pd
from utils.schema_discovery import discover_dataset_schema

# Configure logging for the main application
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [Main-LLM] - %(message)s'
)


def main():
    """
    Main entry point for the LLM-powered MAS application.
    """

    parser = argparse.ArgumentParser(
        description="Run the LLM-powered MAS workflow."
    )

    # =========================================================
    # DEFAULT = OLLAMA
    # =========================================================
    parser.add_argument(
        '--planner-llm',
        type=str,
        default='ollama',
        help="Planner LLM backend: 'ollama', 'gemini', or 'mock'"
    )

    parser.add_argument(
        '--planner-model',
        type=str,
        default=None,
        help="Model name for planner LLM"
    )

    parser.add_argument(
        '--decision-llm',
        type=str,
        default='ollama',
        help="Decision LLM backend"
    )

    parser.add_argument(
        '--decision-model',
        type=str,
        default=None,
        help="Decision model name"
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help="Path to CSV dataset"
    )

    parser.add_argument(
        '--auto',
        action='store_true',
        help="Enable auto mode"
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help="Process all datasets"
    )

    parser.add_argument(
        '--interface',
        type=str,
        default='cli',
        help="HITL interface"
    )

    args = parser.parse_args()

    # =========================================================
    # HITL INTERFACE
    # =========================================================
    from utils.hitl_interface import get_hitl_interface

    hitl_interface = get_hitl_interface(args.interface)

    if args.auto or args.batch:
        os.environ["HITL_AUTO"] = "1"

    # =========================================================
    # DIGITAL MARKETING SCHEMA LOGIC
    # =========================================================
    def auto_select_from_schema(csv_path: str):

        df = pd.read_csv(csv_path)

        schema = discover_dataset_schema(df)

        # =====================================================
        # PRIORITY TARGETS FOR DIGITAL MARKETING
        # =====================================================
        preferred_targets = [
            "Conversion",
            "ConversionRate",
            "ClickThroughRate"
        ]

        target = None
        problem = "classification"

        for col in preferred_targets:

            if col in df.columns:

                target = col

                if col == "Conversion":
                    problem = "classification"
                else:
                    problem = "regression"

                break

        # =====================================================
        # FALLBACK LOGIC
        # =====================================================
        if target is None:

            suggested = schema.get('suggested_targets', []) or []

            if suggested:

                best = sorted(
                    suggested,
                    key=lambda x: x.get('score', 0),
                    reverse=True
                )[0]

                target = best.get('column')

                problem = best.get(
                    'suggested_task',
                    'classification'
                )

            else:

                last_col = df.columns[-1]

                target = last_col

                if (
                    str(df[last_col].dtype)
                    in ['object', 'category']
                    or df[last_col].nunique() <= 20
                ):
                    problem = 'classification'
                else:
                    problem = 'regression'

        # =====================================================
        # FEATURE FILTERING
        # =====================================================
        cols_info = schema.get('columns', {})

        excluded_cols = [
            "CustomerID"
        ]

        # Remove noisy/static columns
        for noisy_col in [
            "AdvertisingPlatform",
            "AdvertisingTool"
        ]:

            if noisy_col in df.columns:

                if df[noisy_col].nunique() <= 1:
                    excluded_cols.append(noisy_col)

        feature_cols = [

            c for c in df.columns

            if c != target
            and c not in excluded_cols
            and cols_info.get(c, {}).get('role')
            not in ['identifier', 'timestamp']
        ]

        logging.info(
            f"[Marketing Mode] Selected Target: {target}"
        )

        logging.info(
            f"[Marketing Mode] Problem Type: {problem}"
        )

        logging.info(
            f"[Marketing Mode] Features: {feature_cols}"
        )

        return feature_cols, target, problem

    # =========================================================
    # RUN SINGLE DATASET
    # =========================================================
    def run_single_dataset(csv_path: str):

        if args.auto:

            feature_cols, target_col, problem_type = (
                auto_select_from_schema(csv_path)
            )

        else:

            feature_cols = None
            target_col = None
            problem_type = None

            (
                dataset_path,
                feature_cols,
                target_col,
                problem_type
            ) = LLMPlannerAgent.interactive_setup(hitl_interface)

            csv_path = dataset_path

        # =====================================================
        # DIGITAL MARKETING GOAL
        # =====================================================
        goal = (
            "Analyze digital marketing campaign data, "
            "predict conversion outcomes, identify key "
            "factors affecting campaign performance, "
            "and generate data-driven advertising "
            "decision recommendations."
        )

        logging.info(
            "Initializing the LLM-powered MAS application..."
        )

        llm_agent = None
        decision_llm_agent = None

        # =====================================================
        # PLANNER LLM
        # =====================================================
        if args.planner_llm == 'ollama':

            from agents.local_llm_agent import LocalLLMAgent

            planner_model = (
                args.planner_model or 'llama3:8b'
            )

            llm_agent = LocalLLMAgent(
                backend='ollama',
                model_name=planner_model
            )

            logging.info(
                f"Using local planner LLM agent: "
                f"ollama, model={planner_model}"
            )

        elif args.planner_llm == 'mock':

            from agents.local_llm_agent import LocalLLMAgent

            llm_agent = LocalLLMAgent(backend='mock')

            logging.info("Using mock planner LLM agent")

        # =====================================================
        # DECISION LLM
        # =====================================================
        if args.decision_llm == 'ollama':

            from agents.local_llm_agent import LocalLLMAgent

            decision_model = (
                args.decision_model or 'llama3:8b'
            )

            decision_llm_agent = LocalLLMAgent(
                backend='ollama',
                model_name=decision_model
            )

            logging.info(
                f"Using local decision LLM agent: "
                f"ollama, model={decision_model}"
            )

        elif args.decision_llm == 'mock':

            from agents.local_llm_agent import LocalLLMAgent

            decision_llm_agent = LocalLLMAgent(
                backend='mock'
            )

            logging.info(
                "Using mock decision LLM agent"
            )

        # =====================================================
        # RUN WORKFLOW
        # =====================================================
        try:

            llm_planner = LLMPlannerAgent(

                dataset_path=csv_path,

                feature_columns=feature_cols,

                target_column=target_col,

                problem_type=problem_type,

                llm_agent=llm_agent,

                decision_llm_agent=decision_llm_agent,

                hitl_interface=hitl_interface
            )

            llm_planner.run_workflow_with_llm(goal)

        except Exception as e:

            logging.error(
                f"An error occurred during the "
                f"LLM workflow: {e}",
                exc_info=True
            )

    # =========================================================
    # BATCH MODE
    # =========================================================
    if args.batch:

        data_dir = os.path.join(
            os.path.dirname(__file__),
            'data'
        )

        all_csvs = []

        for root, dirs, files in os.walk(data_dir):

            for f in files:

                if f.lower().endswith('.csv'):

                    all_csvs.append(
                        os.path.join(root, f)
                    )

        logging.info(
            f"Batch mode: found {len(all_csvs)} datasets"
        )

        for csv in all_csvs:

            logging.info(
                f"--- Processing dataset: {csv} ---"
            )

            run_single_dataset(csv)

    else:

        if args.dataset is None and args.auto:

            logging.error(
                "--auto requires --dataset."
            )

            return

        csv_path = args.dataset

        if not csv_path:

            (
                dataset_path,
                feature_cols,
                target_col,
                problem_type
            ) = LLMPlannerAgent.interactive_setup(
                hitl_interface
            )

            csv_path = dataset_path

        else:

            run_single_dataset(csv_path)

    logging.info(
        "LLM-powered MAS application "
        "has finished its run."
    )


if __name__ == "__main__":
    main()