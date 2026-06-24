"""
main_comparison.py
──────────────────
Menjalankan agentic AI pada dataset yang sama dan menghasilkan laporan analisis.

System: Qwen (local via Ollama) — default model: qwen3:4b

Cara pakai:
    python main_comparison.py --dataset data/your_dataset.csv

Opsi tambahan:
    --qwen-model   qwen3:4b          (model Ollama, default: qwen3:4b)
    --auto                           (skip interactive setup)
    --no-hitl                        (skip HITL review, auto-approve)
"""

import logging
import os
import sys
import argparse
import time
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s'
)

# ──────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Jalankan Agentic AI: Qwen (local via Ollama)"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help="Path ke file CSV dataset. Jika tidak diisi, program akan meminta input."
    )
    parser.add_argument(
        '--qwen-model',
        type=str,
        default='qwen3:4b',
        help="Model Ollama untuk Qwen (default: qwen3:4b)"
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help="Aktifkan auto-mode (skip interactive setup, auto-approve HITL)"
    )
    parser.add_argument(
        '--no-hitl',
        action='store_true',
        help="Skip HITL review: semua rekomendasi otomatis disetujui"
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_dataset(csv_path: str):
    """
    Detect target column and feature columns from the dataset.
    Returns (feature_cols, target_col, problem_type).
    """
    from utils.schema_discovery import discover_dataset_schema

    df = pd.read_csv(csv_path)

    # Priority targets for digital marketing
    preferred_targets = ["Conversion", "ConversionRate", "ClickThroughRate"]
    target = None
    problem = "classification"

    for col in preferred_targets:
        if col in df.columns:
            target = col
            problem = "classification" if col == "Conversion" else "regression"
            break

    if target is None:
        schema = discover_dataset_schema(df)
        suggested = schema.get('suggested_targets', []) or []
        if suggested:
            best = sorted(suggested, key=lambda x: x.get('score', 0), reverse=True)[0]
            target = best.get('column')
            problem = best.get('problem_type', 'classification')
        else:
            raise ValueError(
                "Tidak dapat mendeteksi target kolom secara otomatis. "
                "Pastikan dataset mengandung kolom 'Conversion'."
            )

    excluded_cols = ['CustomerID', 'customer_id', 'ID', 'id',
                     'AdvertisingPlatform', 'AdvertisingTool',
                     'Date', 'date', 'Gender', 'gender']

    schema = discover_dataset_schema(df)
    cols_info = schema.get('columns', {})

    feature_cols = [
        c for c in df.columns
        if c != target
        and c not in excluded_cols
        and cols_info.get(c, {}).get('role') not in ['identifier', 'timestamp']
    ]

    logging.info(f"[Setup] Target: {target} | Problem: {problem}")
    logging.info(f"[Setup] Features ({len(feature_cols)}): {feature_cols}")

    return feature_cols, target, problem


# ──────────────────────────────────────────────────────────────────────────────
# Single Workflow Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_single_workflow(
    system_name: str,
    csv_path: str,
    feature_cols: list,
    target_col: str,
    problem_type: str,
    llm_agent=None,
    hitl_auto: bool = False,
) -> dict:
    """
    Run one full workflow and return metrics.

    Parameters
    ----------
    system_name  : Display name for logs (e.g. "Qwen (local)")
    csv_path     : Path to dataset CSV
    feature_cols : List of feature column names
    target_col   : Target column name
    problem_type : "classification" or "regression"
    llm_agent    : LocalLLMAgent instance
    hitl_auto    : If True, auto-approve all recommendations (no user input)

    Returns
    -------
    dict of metrics from get_comparison_metrics()
    """

    from agents.llm_planner_agent import LLMPlannerAgent
    from utils.hitl_interface import get_hitl_interface

    print()
    print("═" * 70)
    print(f"  🚀  MENJALANKAN: {system_name}")
    print("═" * 70)
    print()

    if hitl_auto:
        os.environ["HITL_AUTO"] = "1"
    else:
        os.environ.pop("HITL_AUTO", None)

    hitl = get_hitl_interface("cli")

    goal = (
        "Analyze digital marketing campaign data, "
        "predict conversion outcomes, identify key "
        "factors affecting campaign performance, "
        "and generate data-driven advertising "
        "decision recommendations."
    )

    try:
        planner = LLMPlannerAgent(
            dataset_path=csv_path,
            feature_columns=feature_cols,
            target_column=target_col,
            problem_type=problem_type,
            llm_agent=llm_agent,
            decision_llm_agent=llm_agent,
            hitl_interface=hitl,
        )

        planner.run_workflow_with_llm(goal)

        metrics = planner.get_comparison_metrics()
        metrics["system_name"] = system_name
        metrics["status"] = "success"

        print()
        print(f"  ✅  {system_name} selesai dalam {metrics.get('duration_seconds', '?')}s")
        return metrics

    except Exception as e:
        logging.error(f"[{system_name}] Workflow gagal: {e}", exc_info=True)
        return {
            "system_name": system_name,
            "status": "failed",
            "error": str(e),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Resolve dataset path ──────────────────────────────────────────────
    csv_path = args.dataset

    if not csv_path:
        # Try default location
        default = os.path.join(
            "data",
            "Digital Marketing Campaign Dataset",
            "digital_marketing_campaign_dataset.csv",
        )
        if os.path.exists(default):
            csv_path = default
            logging.info(f"[Setup] Menggunakan dataset default: {csv_path}")
        else:
            csv_path = input("\n📂 Masukkan path ke file CSV dataset: ").strip()

    if not os.path.exists(csv_path):
        print(f"\n❌  File tidak ditemukan: {csv_path}")
        sys.exit(1)

    # ── Setup dataset schema ──────────────────────────────────────────────
    logging.info("[Setup] Menganalisis skema dataset...")
    feature_cols, target_col, problem_type = setup_dataset(csv_path)

    auto_mode = args.auto or args.no_hitl

    # ──────────────────────────────────────────────────────────────────────
    # SYSTEM — Qwen (local via Ollama)
    # ──────────────────────────────────────────────────────────────────────
    from agents.local_llm_agent import LocalLLMAgent

    qwen_agent = LocalLLMAgent(
        backend='ollama',
        model_name=args.qwen_model
    )

    metrics = run_single_workflow(
        system_name=f"Qwen ({args.qwen_model}) — Local",
        csv_path=csv_path,
        feature_cols=feature_cols,
        target_col=target_col,
        problem_type=problem_type,
        llm_agent=qwen_agent,
        hitl_auto=auto_mode,
    )

    # ──────────────────────────────────────────────────────────────────────
    # REPORT
    # ──────────────────────────────────────────────────────────────────────
    print()
    if metrics.get("status") == "success":
        print("✅  Workflow selesai.")
        print()
        print("  Hasil:")
        for k, v in metrics.items():
            if k not in ("rec_by_area", "rec_by_channel", "rec_by_priority", "system_name"):
                print(f"  {k:<30} {v}")
    else:
        print("❌  Workflow gagal.")
        print(f"   Status: {metrics.get('status')} — {metrics.get('error','')}")


if __name__ == "__main__":
    main()
