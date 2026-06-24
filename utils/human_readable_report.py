"""
human_readable_report.py
========================
Menghasilkan output laporan yang jelas, ringkas, dan dapat dibaca
langsung oleh pengguna non-teknis setelah workflow AI agent selesai.

Menggantikan output log mentah dengan:
  1. Ringkasan performa model (dalam bahasa manusia)
  2. Metrik kampanye (CTR, Conversion Rate, ROI estimasi)
  3. Rekomendasi terstruktur per kategori aksi
  4. Panduan langkah konkret berikutnya
"""

import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _bar(value: float, total: float = 1.0, width: int = 20) -> str:
    """Membuat progress bar ASCII sederhana."""
    filled = int((value / total) * width) if total > 0 else 0
    filled = max(0, min(filled, width))
    return "█" * filled + "░" * (width - filled)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _grade(accuracy: float) -> str:
    if accuracy >= 0.90:
        return "Sangat Baik ✅"
    elif accuracy >= 0.80:
        return "Baik ✅"
    elif accuracy >= 0.70:
        return "Cukup ⚠️"
    elif accuracy >= 0.60:
        return "Lemah ⚠️"
    else:
        return "Tidak Handal ❌"


def _impact_icon(impact: str) -> str:
    icons = {
        "High": "🔴",
        "Medium-High": "🟠",
        "Medium": "🟡",
        "Low": "🟢",
    }
    return icons.get(impact, "⚪")


def _priority_icon(priority: str) -> str:
    icons = {
        "Critical": "🚨",
        "High": "🔥",
        "Growth": "📈",
        "Opportunity": "💡",
        "Advisory": "ℹ️",
    }
    return icons.get(priority, "•")


def _channel_icon(channel: str) -> str:
    icons = {
        "Social Media": "📱",
        "Email": "📧",
        "SEO": "🔍",
        "PPC": "💰",
        "Referral": "🤝",
    }
    return icons.get(channel, "📣")


# ─────────────────────────────────────────────────────────────
# FUNGSI UTAMA
# ─────────────────────────────────────────────────────────────

def print_human_readable_report(
    performance_metrics: Dict[str, Any],
    recommendations_df: Optional[pd.DataFrame],
    feature_importances: Optional[pd.DataFrame] = None,
    dataset_info: Optional[Dict[str, Any]] = None,
    duration_seconds: Optional[float] = None,
):
    """
    Cetak laporan lengkap yang mudah dibaca ke console.
    Dipanggil setelah workflow selesai sebagai pengganti
    logging mentah.
    """
    now = datetime.now().strftime("%d %B %Y, %H:%M")

    print()
    print("  ✅  Analisis selesai — menyusun laporan akhir...")
    print()

    _print_header(now, duration_seconds)
    _print_benchmark_comparison(performance_metrics)
    _print_model_performance(performance_metrics)
    _print_feature_drivers(feature_importances)
    _print_campaign_metrics_summary(recommendations_df)
    _print_recommendations(recommendations_df)
    _print_next_steps(recommendations_df)
    _print_footer()



# ─────────────────────────────────────────────────────────────
# SECTION 0 – TABEL PERBANDINGAN BENCHMARK SEMUA MODEL
# ─────────────────────────────────────────────────────────────

def _print_benchmark_comparison(metrics: Dict[str, Any]):
    """
    Cetak tabel ringkasan benchmark semua model jika tersedia.
    Data berasal dari benchmark_table yang disimpan di performance_metrics.
    """
    benchmark_table = metrics.get("benchmark_table")
    if not benchmark_table:
        return

    # Deteksi task dari kolom yang ada di baris pertama
    first = benchmark_table[0]
    is_classification = "accuracy" in first

    sep_thick = "═" * 76
    sep_thin  = "─" * 76

    print()
    print(sep_thick)
    print("  📊  PERBANDINGAN SEMUA MODEL KLASIFIKASI (BENCHMARK PENELITIAN)")
    print(sep_thick)
    print("  Catatan: Semua model dijalankan pada data yang sama (80% train / 20% test)")
    print()

    best_model = metrics.get("model_used", "")

    if is_classification:
        print(f"  {'No.':<4} {'Model':<34} {'Akurasi':>8} {'Presisi':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}  {'Waktu':>6}")
        print(f"  {sep_thin}")
        for i, row in enumerate(benchmark_table, 1):
            if row.get("status", "").startswith("❌"):
                mark = "❌"
                acc = prec = rec = f1 = auc = "  error"
                t = f"{row.get('elapsed_s', 0):>5.1f}s"
            else:
                mark = "⭐" if row["model"] == best_model else "  "
                acc  = f"{row.get('accuracy',  0)*100:>7.2f}%"
                prec = f"{row.get('precision', 0)*100:>7.2f}%"
                rec  = f"{row.get('recall',    0)*100:>7.2f}%"
                f1   = f"{row.get('f1_score',  0)*100:>7.2f}%"
                auc_val = row.get('roc_auc')
                auc  = f"{auc_val*100:>7.2f}%" if isinstance(auc_val, float) else f"{'N/A':>8}"
                t    = f"{row.get('elapsed_s', 0):>5.1f}s"
            label = row['model']
            if row["model"] == best_model:
                label += " (terbaik)"
            print(f"  {mark} {i:<3} {label:<34} {acc} {prec} {rec} {f1} {auc}  {t}")
    else:
        print(f"  {'No.':<4} {'Model':<38} {'R²':>9} {'MSE':>12} {'RMSE':>9}  {'Waktu':>6}")
        print(f"  {sep_thin}")
        for i, row in enumerate(benchmark_table, 1):
            mark = "⭐" if row["model"] == best_model else "  "
            r2   = f"{row.get('r2',   0):>9.4f}"
            mse  = f"{row.get('mse',  0):>12.4f}"
            rmse = f"{row.get('rmse', 0):>9.4f}"
            t    = f"{row.get('elapsed_s', 0):>5.1f}s"
            label = row["model"]
            if row["model"] == best_model:
                label += " (terbaik)"
            print(f"  {mark} {i:<3} {label:<38} {r2} {mse} {rmse}  {t}")

    print(f"  {sep_thin}")
    print(f"  ⭐ Model yang digunakan untuk rekomendasi: {best_model}")
    print(sep_thick)
    print()


# ─────────────────────────────────────────────────────────────
# SECTION 1 – HEADER
# ─────────────────────────────────────────────────────────────

def _print_header(now: str, duration: Optional[float]):
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       LAPORAN ANALISIS KAMPANYE DIGITAL MARKETING            ║")
    print("║             AI Agent – Ringkasan Eksekutif                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  🗓  Dihasilkan : {now}")
    if duration:
        print(f"  ⏱  Durasi     : {duration:.0f} detik")
    print()


# ─────────────────────────────────────────────────────────────
# SECTION 2 – PERFORMA MODEL
# ─────────────────────────────────────────────────────────────

def _print_model_performance(metrics: Dict[str, Any]):
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  📊  PERFORMA MODEL PREDIKSI KONVERSI")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    model_name = metrics.get("model_used", metrics.get("model", "Unknown"))
    print(f"  Model yang digunakan  : {model_name}")
    print()

    accuracy = metrics.get("accuracy")
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    f1 = metrics.get("f1_score", metrics.get("f1"))
    roc_auc = metrics.get("roc_auc")

    if accuracy is not None:
        grade = _grade(accuracy)
        bar = _bar(accuracy)
        print(f"  Akurasi     {bar}  {_pct(accuracy):>6}  →  {grade}")
        print(f"    ↳ Artinya: dari 100 pelanggan, model benar memprediksi"
              f" ±{accuracy*100:.0f} orang akan / tidak konversi.")
        print()

    if precision is not None:
        bar = _bar(precision)
        print(f"  Presisi     {bar}  {_pct(precision):>6}")
        print(f"    ↳ Artinya: {_pct(precision)} rekomendasi 'konversi' "
              f"yang diberikan model benar-benar terjadi.")
        print()

    if recall is not None:
        bar = _bar(recall)
        print(f"  Recall      {bar}  {_pct(recall):>6}")
        print(f"    ↳ Artinya: model berhasil menangkap {_pct(recall)} "
              f"dari seluruh pelanggan yang benar-benar konversi.")
        print()

    if f1 is not None:
        bar = _bar(f1)
        print(f"  F1-Score    {bar}  {_pct(f1):>6}")
        print(f"    ↳ Keseimbangan antara presisi dan recall. "
              f"Semakin tinggi semakin baik.")
        print()

    if roc_auc is not None:
        bar = _bar(roc_auc)
        print(f"  ROC-AUC     {bar}  {_pct(roc_auc):>6}")
        print(f"    ↳ Kemampuan model membedakan pelanggan yang konversi "
              f"vs tidak. >70% = dapat diandalkan.")
        print()

    # Interpretasi keseluruhan
    if accuracy is not None:
        print("  📌 Kesimpulan Model:")
        if accuracy >= 0.85:
            print("     Model sangat andal untuk mendukung keputusan "
                  "alokasi anggaran kampanye.")
        elif accuracy >= 0.75:
            print("     Model cukup andal. Gunakan bersama penilaian "
                  "tim marketing Anda.")
        else:
            print("     Model masih perlu ditingkatkan. Gunakan rekomendasi "
                  "sebagai panduan awal, bukan keputusan final.")
    print()


# ─────────────────────────────────────────────────────────────
# SECTION 3 – FAKTOR PENGGERAK
# ─────────────────────────────────────────────────────────────

def _print_feature_drivers(feature_importances: Optional[pd.DataFrame]):
    if feature_importances is None or feature_importances.empty:
        return

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  🔎  FAKTOR PALING BERPENGARUH TERHADAP KONVERSI")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  (Faktor-faktor di bawah ini paling menentukan apakah")
    print("   seorang pelanggan akan melakukan pembelian atau tidak)")
    print()

    # Bersihkan nama kolom
    feat_col = "feature" if "feature" in feature_importances.columns else feature_importances.columns[0]
    imp_col  = "importance" if "importance" in feature_importances.columns else feature_importances.columns[1]

    top = feature_importances.nlargest(5, imp_col)
    max_imp = top[imp_col].max()

    # Mapping nama fitur ke bahasa manusia
    label_map = {
        "PagesPerVisit":       "Halaman per Kunjungan",
        "AdSpend":             "Pengeluaran Iklan",
        "LoyaltyPoints":       "Poin Loyalitas Pelanggan",
        "PreviousPurchases":   "Riwayat Pembelian Sebelumnya",
        "EmailClicks":         "Klik Email",
        "ClickThroughRate":    "Click-Through Rate (CTR)",
        "ConversionRate":      "Tingkat Konversi",
        "WebsiteVisits":       "Kunjungan Website",
        "TimeOnSite":          "Durasi di Website",
        "EmailOpens":          "Email Dibuka",
        "Income":              "Pendapatan Pelanggan",
        "Age":                 "Usia Pelanggan",
    }

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        raw_name = str(row[feat_col]).replace("num__", "").replace("cat__", "").replace("remainder__", "")
        label = label_map.get(raw_name, raw_name)
        imp   = row[imp_col]
        bar   = _bar(imp, max_imp, 18)
        pct_str = f"{imp*100:.1f}%"
        print(f"  {rank}. {label:<35} {bar}  {pct_str}")

    print()
    print("  📌 Artinya: Fokuskan optimasi kampanye pada faktor-faktor")
    print("     di atas karena paling besar dampaknya pada konversi.")
    print()


# ─────────────────────────────────────────────────────────────
# SECTION 4 – RINGKASAN METRIK KAMPANYE
# ─────────────────────────────────────────────────────────────

def _print_campaign_metrics_summary(recommendations_df: Optional[pd.DataFrame]):
    if recommendations_df is None or recommendations_df.empty:
        return

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  📣  RINGKASAN STATUS KAMPANYE")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    total = len(recommendations_df)
    n_critical = (recommendations_df["Priority_Level"] == "Critical").sum() if "Priority_Level" in recommendations_df.columns else 0
    n_high     = (recommendations_df["Priority_Level"] == "High").sum() if "Priority_Level" in recommendations_df.columns else 0
    n_growth   = (recommendations_df["Priority_Level"] == "Growth").sum() if "Priority_Level" in recommendations_df.columns else 0

    # Prediksi konversi
    if "Predicted_Conversion" in recommendations_df.columns:
        n_convert     = (recommendations_df["Predicted_Conversion"] == 1).sum()
        n_no_convert  = (recommendations_df["Predicted_Conversion"] == 0).sum()
        conv_rate_pred = n_convert / total if total > 0 else 0
        print(f"  Pelanggan diprediksi KONVERSI      : {n_convert:>4}  orang  ({_pct(conv_rate_pred)})")
        print(f"  Pelanggan diprediksi TIDAK konversi: {n_no_convert:>4}  orang  ({_pct(1-conv_rate_pred)})")
        print()

    # Distribusi channel
    if "CampaignChannel" in recommendations_df.columns:
        print("  Distribusi masalah per saluran kampanye:")
        channel_counts = recommendations_df["CampaignChannel"].value_counts()
        for ch, cnt in channel_counts.items():
            icon = _channel_icon(ch)
            bar  = _bar(cnt, channel_counts.max(), 12)
            print(f"    {icon}  {ch:<15} {bar}  {cnt} kasus")
        print()

    # Jenis tindakan yang dibutuhkan
    if "Decision_Area" in recommendations_df.columns:
        print("  Jenis tindakan yang diperlukan:")
        area_counts = recommendations_df["Decision_Area"].value_counts()
        for area, cnt in area_counts.items():
            print(f"    • {area:<42}: {cnt} kasus")
        print()

    print(f"  Status urgensi:")
    print(f"    🚨 Kritis  (butuh tindakan sekarang): {n_critical} kasus")
    print(f"    🔥 Tinggi  (tindakan dalam 1 minggu): {n_high} kasus")
    if n_growth > 0:
        print(f"    📈 Tumbuh  (kesempatan untuk scale) : {n_growth} kasus")
    print()


# ─────────────────────────────────────────────────────────────
# SECTION 5 – REKOMENDASI DETAIL
# ─────────────────────────────────────────────────────────────

def _print_recommendations(recommendations_df: Optional[pd.DataFrame]):
    if recommendations_df is None or recommendations_df.empty:
        return

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  🎯  REKOMENDASI AKSI PER KATEGORI")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  (Diurutkan dari yang paling mendesak)")
    print()

    if "Decision_Area" not in recommendations_df.columns:
        return

    # Kelompokkan per Decision_Area
    priority_order = {"Critical": 0, "High": 1, "Growth": 2, "Opportunity": 3, "Advisory": 4}
    df_sorted = recommendations_df.copy()
    if "Priority_Level" in df_sorted.columns:
        df_sorted["_rank"] = df_sorted["Priority_Level"].map(priority_order).fillna(9)
        df_sorted = df_sorted.sort_values("_rank")

    groups = df_sorted.groupby("Decision_Area", sort=False)

    for area, group in groups:
        priority = group["Priority_Level"].iloc[0] if "Priority_Level" in group.columns else "High"
        p_icon = _priority_icon(priority)
        n = len(group)

        print(f"  {p_icon}  {area.upper()}  ({n} kasus)")
        print(f"  {'─' * 58}")

        # Tindakan unik (terjemahan ke Bahasa Indonesia)
        _action_translate = {
            "Revise ad creative, audience targeting, and campaign message because predicted conversion is low and CTR is weak.":
                "Revisi materi iklan, targeting audiens, dan pesan kampanye — konversi diprediksi rendah dan CTR lemah.",
            "Improve email CTA, landing page relevance, and offer clarity because users open emails but do not click enough.":
                "Perkuat CTA email, relevansi landing page, dan kejelasan penawaran — pengguna membuka email tapi tidak cukup mengklik.",
            "Launch retargeting campaign and optimize landing page flow because traffic exists but conversion is predicted low.":
                "Luncurkan kampanye retargeting dan optimalkan alur landing page — ada traffic tapi konversi diprediksi rendah.",
            "Reallocate budget or test alternative channel/campaign type because this audience is predicted not to convert.":
                "Alihkan anggaran atau uji saluran/tipe kampanye alternatif — audiens ini diprediksi tidak akan konversi.",
            "Scale budget gradually for this high-performing campaign segment while monitoring cost efficiency.":
                "Naikkan anggaran secara bertahap untuk segmen kampanye terbaik ini sambil memantau efisiensi biaya.",
            "Prioritize retention, loyalty, and upsell campaigns because the customer segment shows strong purchase history.":
                "Prioritaskan kampanye retensi, loyalitas, dan upsell — segmen ini punya riwayat pembelian yang kuat.",
            "Maintain campaign exposure and use this segment as a benchmark for similar audiences.":
                "Pertahankan eksposur kampanye dan jadikan segmen ini patokan untuk audiens serupa.",
        }
        if "Recommended_Action" in group.columns:
            unique_actions = group["Recommended_Action"].unique()
            for action in unique_actions:
                translated = _action_translate.get(action.strip(), action)
                print(f"  → {translated}")
            print()

        # Contoh kasus nyata (maks 3)
        print("  Contoh kasus dari data Anda:")
        sample = group.head(3)
        for i, (_, row) in enumerate(sample.iterrows(), 1):
            ch       = row.get("CampaignChannel", "?")
            ch_icon  = _channel_icon(ch)
            eng      = row.get("Engagement_Status", "")
            timeframe = row.get("Timeframe", "")
            impact   = row.get("Estimated_Impact", "")
            imp_icon = _impact_icon(impact)

            print(f"    [{i}] {ch_icon} Saluran: {ch}")
            if eng:
                print(f"         Kondisi  : {eng}")
            print(f"         Dampak   : {imp_icon} {impact}   ⏰ Waktu: {timeframe}")

            # Tampilkan faktor kunci jika ada
            factors = row.get("Contributing_Factors", "")
            if factors and factors != "Top feature values unavailable.":
                # Sederhanakan tampilan faktor
                factor_parts = [f.strip() for f in factors.split(",")][:3]
                print(f"         Indikator: {', '.join(factor_parts)}")
            print()

        print()

    print()


# ─────────────────────────────────────────────────────────────
# SECTION 6 – LANGKAH KONKRET BERIKUTNYA
# ─────────────────────────────────────────────────────────────

def _print_next_steps(recommendations_df: Optional[pd.DataFrame]):
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  🗺️   PANDUAN LANGKAH BERIKUTNYA")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()

    has_creative   = False
    has_email      = False
    has_budget     = False
    has_retarget   = False

    if recommendations_df is not None and "Decision_Area" in recommendations_df.columns:
        areas = recommendations_df["Decision_Area"].tolist()
        has_creative = any("Creative" in a for a in areas)
        has_email    = any("Email" in a for a in areas)
        has_budget   = any("Budget" in a for a in areas)
        has_retarget = any("Retarget" in a or "Landing" in a for a in areas)

    step = 1

    if has_creative:
        print(f"  {step}. 🎨  KREATIF & TARGETING  (Lakukan Hari Ini)")
        print("       • Audit materi iklan (gambar, teks, CTA) yang CTR-nya rendah")
        print("       • Uji 2–3 variasi kreatif baru dengan A/B testing")
        print("       • Sempurnakan segmen audiens: usia, lokasi, minat")
        print()
        step += 1

    if has_email:
        print(f"  {step}. 📧  EMAIL MARKETING  (Lakukan dalam 3–7 Hari)")
        print("       • Perkuat tombol CTA di email (warna, teks, posisi)")
        print("       • Pastikan landing page relevan dengan isi email")
        print("       • Uji waktu pengiriman email yang berbeda")
        print()
        step += 1

    if has_retarget:
        print(f"  {step}. 🔁  RETARGETING & LANDING PAGE  (Lakukan dalam 1 Minggu)")
        print("       • Buat kampanye retargeting untuk pengunjung yang belum beli")
        print("       • Sederhanakan alur landing page (kurangi langkah checkout)")
        print("       • Tambah social proof (ulasan, testimoni) di landing page")
        print()
        step += 1

    if has_budget:
        print(f"  {step}. 💰  ALOKASI ANGGARAN  (Evaluasi Minggu Ini)")
        print("       • Hentikan sementara kampanye dengan CTR < 5% dan konversi 0")
        print("       • Pindahkan anggaran ke saluran dengan performa terbaik")
        print("       • Uji saluran/tipe kampanye alternatif untuk segmen yang stagnan")
        print()
        step += 1

    print(f"  {step}. 📅  MONITORING & EVALUASI  (Rutin)")
    print("       • Pantau CTR, konversi, dan ROI setiap 7 hari")
    print("       • Jalankan ulang analisis AI ini setelah 30 hari untuk update prediksi")
    print("       • Dokumentasikan hasil A/B testing sebagai referensi kampanye berikutnya")
    print()


# ─────────────────────────────────────────────────────────────
# SECTION 7 – FOOTER
# ─────────────────────────────────────────────────────────────

def _print_footer():
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  ✅  Laporan selesai. Detail lengkap tersedia di folder logs/")
    print("      (file JSON & CSV untuk analisis lebih lanjut)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
