# Framework Evaluasi Agentic AI untuk Analisa dan Pengambilan Keputusan Digital Marketing

Dokumen ini menjelaskan upgrade arsitektur agar sistem tidak hanya terlihat sebagai pipeline machine learning dengan LLM orchestrator, tetapi sebagai **framework evaluasi Agentic AI** yang memiliki trajectory, benchmark, governance, safety, robustness, dan audit layer.

## 1. Prinsip Framework

Framework ini mengikuti tiga dimensi besar evaluasi agentic AI: **benchmarks, metrics, dan governance**. Sistem tidak hanya mengukur akurasi model, tetapi juga mengevaluasi bagaimana agent mempersepsi data, merencanakan langkah, memilih tool, melakukan tindakan, mengamati hasil, merefleksikan hasil, beradaptasi, dan menjaga kontrol manusia.

## 2. Komponen Baru

### `utils/agentic_evaluation.py`
Modul inti evaluasi Agentic AI. Isinya:

- `AgenticTrajectoryLogger`: mencatat trajectory dalam format `thought/action/observation/reflection` ke JSONL.
- `PlanningEvaluator`: menilai kelengkapan rencana, urutan tool, redundansi, dan kegagalan step.
- `ToolSelectionEvaluator`: mengevaluasi validitas tool dan apakah precondition tool sudah terpenuhi.
- `RobustnessTester`: menjalankan lightweight robustness testing terhadap data dan model.
- `GovernanceSafetyEvaluator`: mengecek HITL, audit log, trajectory log, potensi PII, dan risiko teks rekomendasi.
- `MultiAgentEvaluator`: menilai coverage agent role dan handoff antar-agent.
- `BenchmarkSystem`: mengevaluasi benchmark model, workflow coverage, dan robustness.
- `AgenticEvaluator`: menggabungkan semua dimensi menjadi `overall_agentic_score`.
- `AdaptiveWorkflowController`: fallback state-aware agar workflow adaptif, bukan sekadar linear.
- `AgenticReportGenerator`: menyimpan laporan JSON, Markdown, dan chart PNG.

## 3. Metrics Agentic AI yang Ditambahkan

Sistem sekarang menghasilkan metrik berikut:

| Metric | Makna |
|---|---|
| `agentic_overall_score` | skor keseluruhan agentic AI 0–100 |
| `planning_score` | kualitas rencana dan urutan workflow |
| `tool_selection_score` | ketepatan pemilihan tool dan precondition |
| `reasoning_trace_score` | kelengkapan thought/action/observation/reflection |
| `robustness_score` | stabilitas model/data terhadap perturbasi ringan |
| `governance_safety_score` | kekuatan kontrol HITL, audit, safety, dan privacy guard |
| `benchmark_score` | coverage benchmark model + workflow + robustness |
| `multi_agent_score` | coverage role agent dan handoff |
| `autonomy_control_score` | keseimbangan autonomy, fallback, dan completion |

## 4. Reasoning Trace

Setiap tool execution sekarang direkam sebagai trajectory:

```json
{
  "step_number": 1,
  "agent_role": "perception_agent",
  "thought": "Need perception before any decision.",
  "action": "load_and_inspect_data",
  "observation": "Data loaded. Shape: ...",
  "reflection": "This creates an auditable perception step.",
  "success": true,
  "decision_source": "llm"
}
```

Catatan: field `thought` adalah ringkasan alasan yang aman dan eksplisit untuk kebutuhan audit, bukan dump chain-of-thought internal model.

## 5. Robustness Testing

Robustness tester menjalankan:

1. **row order invariance**: prediksi tidak boleh berubah karena urutan baris berubah.
2. **small numeric noise**: prediksi diuji dengan noise kecil pada fitur numerik.
3. **missing value recovery**: sebagian nilai numerik dibuat missing lalu diimputasi median.
4. **data quality scan**: missing ratio, duplicate ratio, dan extreme outlier ratio.

## 6. Governance & Safety Evaluation

Governance layer mengecek:

- apakah HITL review tercatat;
- apakah file audit HITL ada;
- apakah file trajectory agent ada;
- potensi kolom PII seperti email, phone, address, name, NIK/KTP;
- teks rekomendasi yang mengandung frasa berisiko seperti manipulasi, misleading, diskriminasi, dark pattern, dan sebagainya.

## 7. Adaptive Workflow

Jika LLM gagal menghasilkan JSON valid atau memilih tool yang salah, workflow tidak langsung berhenti. `AdaptiveWorkflowController` membaca state saat ini:

- data sudah diload atau belum;
- data sudah dipreprocess atau belum;
- analisis sudah selesai atau belum;
- rekomendasi sudah dibuat atau belum.

Kemudian controller memilih next action berbasis state sehingga sistem tetap runnable dan adaptif.

## 8. Output Baru

Setelah workflow selesai, sistem menghasilkan:

- `logs/agentic_trajectory_<timestamp>.jsonl`
- `logs/agentic_evaluation_report_<timestamp>.json`
- `logs/agentic_evaluation_report_<timestamp>.md`
- `logs/agentic_evaluation_scores_<timestamp>.png` jika matplotlib tersedia
- laporan utama tetap: `digital_marketing_workflow_report_*.json`
- publication snapshot tetap: `publication_snapshot_*.json` dan CSV rekomendasi

## 9. Cara Menjalankan

Mode offline/mock untuk memastikan runnable:

```bash
python main_llm.py --dataset data.csv --planner-llm mock --decision-llm mock --auto
```

Mode Ollama:

```bash
python main_llm.py --dataset data.csv --planner-llm ollama --planner-model qwen3:4b --decision-llm ollama --decision-model qwen3:4b --auto
```

Mode jalankan Qwen local:

```bash
python main_comparison.py --dataset data.csv --qwen-model qwen3:4b --auto
```

## 10. File yang Diubah

- `agents/llm_planner_agent.py`: integrasi trajectory, agentic evaluation, adaptive workflow, HITL auto/non-interactive, fallback ke Qwen local (Ollama).
- `agents/dynamic_analysis_agent.py`: hasil analisis menyimpan fitted model agar robustness testing bisa dijalankan.
- `utils/reporting.py`: report utama sekarang menyimpan hasil agentic evaluation.
- `utils/comparison_report.py`: comparison report sekarang membandingkan model performance dan skor agentic AI.
- `utils/agentic_evaluation.py`: modul baru untuk metrics, governance, robustness, benchmark, trajectory, dan reporting.
- `utils/__init__.py`: memastikan folder utils menjadi package eksplisit.

## 11. Kenapa Ini Lebih Agentic

Sebelumnya sistem sudah punya multi-agent workflow, tetapi evaluasinya dominan pada hasil model. Setelah upgrade, objek penelitian bergeser menjadi **perilaku agentic system**: bagaimana agent merencanakan, memilih tool, berinteraksi dengan environment, mencatat trajectory, menguji robustness, menjalankan governance, melibatkan HITL, dan menghasilkan laporan audit.
