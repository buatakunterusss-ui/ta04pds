"""
Simulasi Monte Carlo: Probabilitas Penurunan Pendapatan Kampanye Iklan Digital
TA-03 Pemodelan dan Simulasi (versi Flask)
Jalankan lokal: python app.py  ->  http://127.0.0.1:5000
"""
import base64
import glob
import io
import os

import matplotlib

matplotlib.use("Agg")  # backend tanpa GUI, wajib untuk server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_cache = {}


# ---------------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def get_data_params():
    """Baca CSV di folder data/, bersihkan outlier, hitung parameter (di-cache)."""
    if "params" in _cache:
        return _cache["params"]

    files = glob.glob(os.path.join(BASE_DIR, "data", "*.csv"))
    if not files:
        raise FileNotFoundError("Taruh file CSV dataset di folder data/")
    df = pd.read_csv(files[0])
    if "ConversionRate" not in df.columns:
        raise ValueError("Dataset harus punya kolom ConversionRate")

    # Cleaning outlier ekstrem (3 x IQR)
    q1, q3 = df["ConversionRate"].quantile([0.25, 0.75])
    iqr = q3 - q1
    clean = df[(df["ConversionRate"] >= q1 - 3 * iqr) &
               (df["ConversionRate"] <= q3 + 3 * iqr)]
    cr = clean["ConversionRate"]
    mu, sigma = cr.mean(), cr.std()

    # Histogram data asli
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cr, bins=40, density=True, alpha=0.6, edgecolor="black", label="Data asli")
    xs = np.linspace(cr.min(), cr.max(), 200)
    ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)),
            "r-", label="Kurva Normal")
    ax.set_title("Distribusi ConversionRate (Data Asli)")
    ax.set_xlabel("ConversionRate")
    ax.set_ylabel("Densitas")
    ax.legend()

    _cache["params"] = {
        "filename": os.path.basename(files[0]),
        "n_raw": len(df), "n_clean": len(clean),
        "mu": mu, "sigma": sigma, "lo": cr.min(), "hi": cr.max(),
        "hist_img": fig_to_base64(fig),
    }
    return _cache["params"]


def run_simulation(p, n_sim, hari, visitors, nilai, dist, seed):
    """Tiap baris = 1 skenario, tiap kolom = 1 hari."""
    np.random.seed(seed)
    if dist == "normal":
        cr = np.random.normal(p["mu"], p["sigma"], size=(n_sim, hari))
    else:
        cr = np.random.uniform(p["lo"], p["hi"], size=(n_sim, hari))
    cr = np.clip(cr, 0, None)  # conversion rate tidak boleh negatif
    return (visitors * cr * nilai).sum(axis=1)


def num(name, default, lo, hi, cast=float):
    """Ambil angka dari form, batasi rentangnya agar aman."""
    try:
        val = cast(request.form.get(name, default))
    except (TypeError, ValueError):
        val = default
    return max(lo, min(hi, val))


# ---------------------------------------------------------------
# Route
# ---------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    try:
        p = get_data_params()
    except Exception as e:  # tampilkan pesan ramah jika data bermasalah
        return render_template("index.html", error=str(e))

    form = {"seed": 301230043, "n_sim": 10000, "hari": 30, "pct": 10,
            "visitors": 1000, "nilai": 50, "dist": "normal"}
    result = None

    if request.method == "POST":
        form = {
            "seed": num("seed", 301230043, 0, 2**32 - 1, int),
            "n_sim": num("n_sim", 10000, 1000, 100000, int),
            "hari": num("hari", 30, 1, 365, int),
            "pct": num("pct", 10, 1, 50, float),
            "visitors": num("visitors", 1000, 1, 1_000_000, int),
            "nilai": num("nilai", 50, 0.01, 1_000_000, float),
            "dist": "uniform" if request.form.get("dist") == "uniform" else "normal",
        }
        total = run_simulation(p, form["n_sim"], form["hari"], form["visitors"],
                               form["nilai"], form["dist"], form["seed"])
        baseline = form["visitors"] * p["mu"] * form["nilai"] * form["hari"]
        threshold = baseline * (1 - form["pct"] / 100)
        prob = float(np.mean(total < threshold))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(total, bins=60, alpha=0.7, edgecolor="black")
        ax.axvline(threshold, color="red", linestyle="--",
                   label=f"Threshold (-{form['pct']:g}%) = {threshold:,.0f}")
        ax.axvline(baseline, color="green", linestyle="-", label=f"Baseline = {baseline:,.0f}")
        ax.set_title(f"{form['n_sim']:,} skenario | P(turun > {form['pct']:g}%) = {prob:.2%}")
        ax.set_xlabel(f"Total pendapatan {form['hari']} hari")
        ax.set_ylabel("Frekuensi")
        ax.legend()

        level = "RENDAH" if prob < 0.05 else ("SEDANG" if prob < 0.25 else "TINGGI")
        result = {
            "prob": prob, "baseline": baseline, "threshold": threshold,
            "mean": float(total.mean()), "var5": float(np.percentile(total, 5)),
            "level": level, "img": fig_to_base64(fig),
            "sens": [(f"> {s}%", float(np.mean(total < baseline * (1 - s / 100))))
                     for s in (5, 10, 15, 20)],
        }

    return render_template("index.html", p=p, form=form, result=result, error=None)


if __name__ == "__main__":
    app.run(debug=True)