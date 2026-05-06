"""
sentiment_finbert.py - Analyse de sentiment financier avec FinBERT + Yahoo Finance
Auteure : Vanelle Stephanie MANGOUA DJOUSSEU

Architecture : ProsusAI/finbert (BERT fine-tune sur textes financiers)
Cas d'usage  : analyse de news financieres + correlation avec cours boursiers reels

Modes :
  - Mode DEMO    (defaut) : classifieur lexical heuristique, sans telechargement
  - Mode FINBERT          : vrai modele HuggingFace ProsusAI/finbert (~500MB)
    python sentiment_finbert.py --mode finbert

Donnees marche : Yahoo Finance API (yfinance) — cours reels BNP.PA, GLE.PA, ACA.PA...
"""

import argparse
import json
import os
import time
import random
import warnings
from datetime import datetime, timedelta
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Corpus de phrases financieres synthetiques (pour mode demo)
# --------------------------------------------------------------------------

PHRASES_POSITIVES = [
    "The company reported record quarterly earnings, beating analyst expectations.",
    "Strong revenue growth driven by robust demand across all business segments.",
    "The board approved a significant dividend increase, rewarding shareholders.",
    "Acquisition of the startup is expected to accelerate digital transformation.",
    "Cost reduction initiatives led to a substantial improvement in operating margin.",
    "The firm secured a major long-term contract worth 500 million euros.",
    "Credit rating upgraded to AA+ following consecutive years of profitability.",
    "New product launch exceeded initial sales targets by 35 percent.",
    "The bank reported a sharp decline in non-performing loans.",
    "Operating cash flow reached an all-time high of 2.3 billion euros.",
    "Investor confidence soared after the CEO outlined a bold growth strategy.",
    "Net interest margin widened, boosting profitability for the financial institution.",
]

PHRASES_NEGATIVES = [
    "The company missed earnings estimates for the third consecutive quarter.",
    "Regulatory fines totaling 1.2 billion euros were imposed by the European authority.",
    "Rising interest rates are putting pressure on the bank's net interest income.",
    "The firm announced a major restructuring plan, cutting 4,000 jobs worldwide.",
    "Credit losses surged as default rates climbed among retail borrowers.",
    "The stock plunged 18 percent following a profit warning from management.",
    "Loan loss provisions increased sharply amid deteriorating macroeconomic conditions.",
    "The audit committee flagged material weaknesses in internal financial controls.",
    "Currency headwinds significantly impacted revenue reported in euros.",
    "The bank faces mounting legal costs related to historical mis-selling claims.",
    "Market share declined as competition intensified in the core retail segment.",
    "Rising inflation eroded consumer purchasing power, dampening loan demand.",
]

PHRASES_NEUTRES = [
    "The central bank kept interest rates unchanged at its quarterly meeting.",
    "Management will host an investor day to outline the five-year strategic plan.",
    "The company completed the acquisition as announced in the prior quarter.",
    "Annual report figures were restated to reflect updated accounting standards.",
    "The board appointed a new Chief Risk Officer effective from January.",
    "Loan volumes remained stable compared to the previous reporting period.",
    "The firm is reviewing strategic options for its non-core asset portfolio.",
    "Quarterly results will be published on the scheduled date in November.",
    "The regulatory sandbox application is currently under review by the authority.",
    "Management reiterated full-year guidance without revision.",
    "The merger is subject to standard antitrust review by competition authorities.",
    "The company maintained its current credit facility with no modifications.",
]

LABELS_STR = {0: "positive", 1: "negative", 2: "neutral"}
COLORS     = {"positive": "#2ca02c", "negative": "#d62728", "neutral": "#1f77b4"}


# --------------------------------------------------------------------------
# Mode DEMO : simuler FinBERT avec un classifieur base sur des regles
# --------------------------------------------------------------------------

MOTS_POS = {
    "record", "growth", "strong", "profit", "increase", "upgrade",
    "exceed", "secured", "dividend", "improvement", "soared",
    "all-time", "high", "beat", "boost", "confidence", "bold",
}
MOTS_NEG = {
    "missed", "fine", "pressure", "restructuring", "default", "surge",
    "plunged", "warning", "loss", "weakness", "declined", "eroded",
    "deteriorating", "mounting", "cut", "penalty", "fraud", "risk",
}


def predict_demo(texte):
    """
    Simule FinBERT via comptage de termes positifs / negatifs.
    Retourne (label, prob_positive, prob_negative, prob_neutral).
    """
    words = set(texte.lower().split())
    pos_count = len(words & MOTS_POS)
    neg_count = len(words & MOTS_NEG)

    base_pos = 0.2 + 0.15 * pos_count
    base_neg = 0.2 + 0.15 * neg_count
    base_neu = 0.6 - 0.05 * (pos_count + neg_count)
    base_neu = max(base_neu, 0.05)

    total = base_pos + base_neg + base_neu
    probs = np.array([base_pos, base_neg, base_neu]) / total
    # Ajouter un peu de bruit
    noise = np.random.dirichlet([10, 10, 10]) * 0.1
    probs = probs * 0.9 + noise
    probs /= probs.sum()

    label_idx = int(np.argmax(probs))
    label_map = {0: "positive", 1: "negative", 2: "neutral"}
    return label_map[label_idx], float(probs[0]), float(probs[1]), float(probs[2])


# --------------------------------------------------------------------------
# Mode FINBERT : vrai modele HuggingFace
# --------------------------------------------------------------------------

def charger_finbert():
    try:
        from transformers import pipeline
        print("[FINBERT] Chargement du modele ProsusAI/finbert ...")
        t0 = time.time()
        nlp = pipeline("text-classification", model="ProsusAI/finbert",
                        return_all_scores=True)
        print("[FINBERT] Modele charge en {:.1f}s".format(time.time() - t0))
        return nlp
    except ImportError:
        print("[ERREUR] Installer : pip install transformers torch")
        return None


def predict_finbert(nlp, texte):
    resultats = nlp(texte[:512])[0]
    scores = {r["label"].lower(): r["score"] for r in resultats}
    label  = max(scores, key=scores.get)
    return (label,
            scores.get("positive", 0.0),
            scores.get("negative", 0.0),
            scores.get("neutral",  0.0))


# --------------------------------------------------------------------------
# Construction du dataset
# --------------------------------------------------------------------------

def construire_dataset(n_par_classe=40, seed=42):
    random.seed(seed)
    rows = []

    for texte in random.sample(PHRASES_POSITIVES * 4, min(n_par_classe, len(PHRASES_POSITIVES) * 4)):
        rows.append({"texte": texte, "label_reel": "positive"})
    for texte in random.sample(PHRASES_NEGATIVES * 4, min(n_par_classe, len(PHRASES_NEGATIVES) * 4)):
        rows.append({"texte": texte, "label_reel": "negative"})
    for texte in random.sample(PHRASES_NEUTRES * 4, min(n_par_classe, len(PHRASES_NEUTRES) * 4)):
        rows.append({"texte": texte, "label_reel": "neutral"})

    # Ajouter des timestamps simules (30 derniers jours)
    base_date = datetime.now()
    for i, row in enumerate(rows):
        delta = timedelta(days=random.randint(0, 29), hours=random.randint(0, 23))
        row["date"] = (base_date - delta).strftime("%Y-%m-%d")
        row["source"] = random.choice(["Reuters", "Bloomberg", "Les Echos",
                                        "Financial Times", "BFM Business"])
        row["ticker"] = random.choice(["BNP.PA", "GLE.PA", "ACA.PA",
                                        "ENGI.PA", "OR.PA", "CAC40"])

    random.shuffle(rows)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Analyse et predictions
# --------------------------------------------------------------------------

def analyser(df, mode="demo", nlp=None):
    print("[ANALYSE] {} phrases en mode {}...".format(len(df), mode.upper()))
    predictions = []
    for _, row in df.iterrows():
        if mode == "finbert" and nlp is not None:
            label, p_pos, p_neg, p_neu = predict_finbert(nlp, row["texte"])
        else:
            label, p_pos, p_neg, p_neu = predict_demo(row["texte"])
        predictions.append({
            "label_predit": label,
            "prob_positive": round(p_pos, 4),
            "prob_negative": round(p_neg, 4),
            "prob_neutral":  round(p_neu, 4),
            "confiance":     round(max(p_pos, p_neg, p_neu), 4),
        })

    df_pred = pd.concat([df.reset_index(drop=True),
                          pd.DataFrame(predictions)], axis=1)
    return df_pred


# --------------------------------------------------------------------------
# Metriques
# --------------------------------------------------------------------------

def calculer_metriques(df):
    from sklearn.metrics import classification_report, confusion_matrix
    labels = ["positive", "negative", "neutral"]
    y_true = df["label_reel"]
    y_pred = df["label_predit"]

    print("\n[METRIQUES] Rapport de classification :")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    accuracy = (y_true == y_pred).mean()
    print("Accuracy globale : {:.2%}".format(accuracy))
    return accuracy


# --------------------------------------------------------------------------
# Yahoo Finance — cours boursiers reels
# --------------------------------------------------------------------------

TICKERS_MARCHE = ["BNP.PA", "GLE.PA", "ACA.PA", "ENGI.PA", "OR.PA"]
TICKER_LABELS  = {"BNP.PA": "BNP Paribas", "GLE.PA": "Soc. Generale",
                  "ACA.PA": "Credit Agricole", "ENGI.PA": "Engie", "OR.PA": "L'Oreal"}


def fetch_cours_boursiers(period="30d"):
    """
    Recupere les cours de cloture reels via Yahoo Finance (yfinance).
    Retourne un DataFrame de rendements journaliers par ticker.
    """
    try:
        import yfinance as yf
        data = yf.download(TICKERS_MARCHE, period=period, interval="1d",
                           progress=False, auto_adjust=True)
        closes = data["Close"].dropna(how="all")
        returns = closes.pct_change().dropna()
        print("[MARKET] Cours reels telecharges : {} jours x {} tickers".format(
            len(returns), len(returns.columns)))
        return returns
    except Exception as e:
        print("[MARKET] yfinance indisponible ({}) — simulation activee".format(e))
        return None


def calculer_score_sentiment_journalier(df):
    """
    Calcule le score de sentiment moyen par jour et par ticker.
    Score = prob_positive - prob_negative  (dans [-1, +1])
    """
    df = df.copy()
    df["score"] = df["prob_positive"] - df["prob_negative"]
    # Exclure CAC40 (indice, pas de cours direct dans les tickers)
    df_titres = df[df["ticker"] != "CAC40"].copy()
    pivot = df_titres.groupby(["date", "ticker"])["score"].mean().reset_index()
    return pivot


def correlation_sentiment_rendement(df_sentiment_pivot, returns):
    """
    Pour chaque ticker, calcule la correlation de Pearson entre :
      - score de sentiment moyen du jour J
      - rendement boursier du jour J+1 (signal predictif)
    Retourne un DataFrame de correlations.
    """
    corrs = []
    for ticker in TICKERS_MARCHE:
        if ticker not in returns.columns:
            continue
        sent_t = df_sentiment_pivot[df_sentiment_pivot["ticker"] == ticker].copy()
        sent_t["date"] = pd.to_datetime(sent_t["date"])
        sent_t = sent_t.set_index("date")["score"]

        ret_t = returns[ticker].copy()
        ret_t.index = pd.to_datetime(ret_t.index)
        # Aligner J et J+1
        ret_shifted = ret_t.shift(-1)
        aligned = pd.concat([sent_t, ret_shifted], axis=1, join="inner")
        aligned.columns = ["sentiment", "return_j1"]
        aligned = aligned.dropna()
        if len(aligned) >= 3:
            corr = aligned["sentiment"].corr(aligned["return_j1"])
            corrs.append({"ticker": ticker, "correlation": corr, "n_obs": len(aligned)})

    return pd.DataFrame(corrs)


# --------------------------------------------------------------------------
# Visualisations
# --------------------------------------------------------------------------

def visualiser(df, returns=None):
    """
    6 panneaux :
      1. Distribution des sentiments predits
      2. Matrice de confusion
      3. Distribution des scores de confiance
      4. Evolution temporelle du sentiment (cours reels superposes si disponibles)
      5. Sentiment moyen par valeur boursiere
      6. Correlation sentiment J / rendement J+1 (Yahoo Finance, si disponible)
    """
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Analyse Sentiment Financier — FinBERT + Yahoo Finance",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    labels   = ["positive", "negative", "neutral"]
    couleurs = [COLORS[l] for l in labels]

    # (1) Distribution des sentiments predits
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["label_predit"].value_counts().reindex(labels, fill_value=0)
    ax1.bar(labels, counts, color=couleurs)
    ax1.set_title("Distribution des sentiments")
    ax1.set_ylabel("Nombre de phrases")
    for i, v in enumerate(counts):
        ax1.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # (2) Matrice de confusion
    ax2 = fig.add_subplot(gs[0, 1])
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm   = confusion_matrix(df["label_reel"], df["label_predit"], labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax2, colorbar=False, cmap="Blues")
    ax2.set_title("Matrice de confusion")
    plt.setp(ax2.get_xticklabels(), rotation=15, fontsize=8)

    # (3) Distribution des scores de confiance
    ax3 = fig.add_subplot(gs[0, 2])
    for lbl, col in COLORS.items():
        subset = df[df["label_predit"] == lbl]["confiance"]
        if len(subset):
            ax3.hist(subset, bins=15, alpha=0.6, label=lbl, color=col)
    ax3.set_xlabel("Score de confiance")
    ax3.set_ylabel("Frequence")
    ax3.set_title("Distribution des scores de confiance")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # (4) Evolution temporelle du sentiment + cours BNP.PA si disponible
    ax4 = fig.add_subplot(gs[1, 0:2])
    df_time = df.groupby(["date", "label_predit"]).size().unstack(fill_value=0)
    for lbl in labels:
        if lbl in df_time.columns:
            ax4.plot(df_time.index, df_time[lbl], marker="o",
                     label=lbl, color=COLORS[lbl], linewidth=2)
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Mentions sentiment")
    ax4.set_title("Evolution temporelle du sentiment")
    ax4.legend(loc="upper left", fontsize=8)
    ax4.tick_params(axis="x", rotation=30)
    ax4.grid(alpha=0.3)

    # Superposer le cours BNP.PA normalise si donnees disponibles
    if returns is not None and "BNP.PA" in returns.columns:
        ax4b = ax4.twinx()
        bnp_cum = (1 + returns["BNP.PA"]).cumprod()
        bnp_cum.index = bnp_cum.index.astype(str)
        # Filtrer sur les dates du corpus
        dates_corpus = sorted(df["date"].unique())
        bnp_filtered = bnp_cum[bnp_cum.index.isin(dates_corpus)]
        if len(bnp_filtered):
            ax4b.plot(bnp_filtered.index, bnp_filtered.values,
                      color="black", linewidth=1.5, linestyle="--",
                      alpha=0.6, label="BNP.PA (rendement cumule)")
            ax4b.set_ylabel("Rendement cumule BNP.PA", fontsize=8)
            ax4b.legend(loc="upper right", fontsize=7)

    # (5) Sentiment net moyen par valeur boursiere
    ax5 = fig.add_subplot(gs[1, 2])
    df_copy = df.copy()
    df_copy["score_net"] = df_copy["prob_positive"] - df_copy["prob_negative"]
    score_par_ticker = df_copy[df_copy["ticker"] != "CAC40"].groupby("ticker")["score_net"].mean()
    colors_bar = ["#2ca02c" if v >= 0 else "#d62728" for v in score_par_ticker.values]
    ax5.barh(score_par_ticker.index, score_par_ticker.values, color=colors_bar)
    ax5.axvline(0, color="black", linewidth=0.8)
    ax5.set_xlabel("Score sentiment net (positif - negatif)")
    ax5.set_title("Sentiment net moyen\npar valeur (30 jours)")
    ax5.grid(axis="x", alpha=0.3)

    out = "sentiment_rapport.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("[VIZ] Rapport sauvegarde : {}".format(out))


def visualiser_correlation(df, returns):
    """
    Panneau supplementaire : correlation sentiment J -> rendement J+1.
    Sauvegarde dans sentiment_correlation.png
    """
    if returns is None:
        return

    df_pivot = calculer_score_sentiment_journalier(df)
    df_corr  = correlation_sentiment_rendement(df_pivot, returns)

    if df_corr.empty:
        print("[MARKET] Pas assez de donnees pour la correlation.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Correlation Sentiment FinBERT / Rendement Boursier (J+1)\n"
                 "Source cours : Yahoo Finance — Valeurs CAC40",
                 fontsize=12, fontweight="bold")

    # Barres de correlation
    ax = axes[0]
    colors = ["#2ca02c" if c >= 0 else "#d62728" for c in df_corr["correlation"]]
    labels_tick = [TICKER_LABELS.get(t, t) for t in df_corr["ticker"]]
    ax.barh(labels_tick, df_corr["correlation"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlation de Pearson (sentiment J, rendement J+1)")
    ax.set_title("Pouvoir predictif du sentiment\nsur le rendement du lendemain")
    ax.grid(axis="x", alpha=0.3)
    for i, (corr, n) in enumerate(zip(df_corr["correlation"], df_corr["n_obs"])):
        ax.text(corr + 0.01 * np.sign(corr), i, "r={:.2f} (n={})".format(corr, n),
                va="center", fontsize=8)

    # Scatter sentiment vs rendement pour BNP.PA
    ax2 = axes[1]
    if "BNP.PA" in returns.columns:
        sent_bnp = df_pivot[df_pivot["ticker"] == "BNP.PA"].copy()
        sent_bnp["date"] = pd.to_datetime(sent_bnp["date"])
        sent_bnp = sent_bnp.set_index("date")["score"]
        ret_bnp = returns["BNP.PA"].shift(-1)
        ret_bnp.index = pd.to_datetime(ret_bnp.index)
        aligned = pd.concat([sent_bnp, ret_bnp], axis=1, join="inner").dropna()
        aligned.columns = ["sentiment", "return_j1"]
        if len(aligned) >= 3:
            ax2.scatter(aligned["sentiment"], aligned["return_j1"] * 100,
                        color="#1f77b4", alpha=0.7, s=50)
            # Droite de regression
            z = np.polyfit(aligned["sentiment"], aligned["return_j1"] * 100, 1)
            p = np.poly1d(z)
            xs = np.linspace(aligned["sentiment"].min(), aligned["sentiment"].max(), 50)
            ax2.plot(xs, p(xs), "r--", alpha=0.6, label="Regression lineaire")
            corr_val = aligned["sentiment"].corr(aligned["return_j1"])
            ax2.set_xlabel("Score sentiment net (FinBERT)")
            ax2.set_ylabel("Rendement J+1 (%)")
            ax2.set_title("BNP.PA — Sentiment J vs Rendement J+1\n(r={:.2f})".format(corr_val))
            ax2.axhline(0, color="black", linewidth=0.5, alpha=0.5)
            ax2.axvline(0, color="black", linewidth=0.5, alpha=0.5)
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = "sentiment_correlation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("[VIZ] Correlation sauvegardee : {}".format(out))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse sentiment financier")
    parser.add_argument("--mode", choices=["demo", "finbert"], default="demo",
                        help="demo=regle heuristique | finbert=vrai modele HuggingFace")
    parser.add_argument("--n",    type=int, default=40,
                        help="Nombre de phrases par classe (defaut: 40)")
    args = parser.parse_args()

    print("=" * 60)
    print("   ANALYSE SENTIMENT FINANCIER - Pipeline FinBERT")
    print("   Mode : {}".format(args.mode.upper()))
    print("=" * 60)

    # Charger FinBERT si necessaire
    nlp = None
    if args.mode == "finbert":
        nlp = charger_finbert()
        if nlp is None:
            print("[FALLBACK] Mode demo active")
            args.mode = "demo"

    # Dataset
    df = construire_dataset(n_par_classe=args.n)
    print("[DATA] Dataset : {} phrases (positives/negatives/neutres)".format(len(df)))

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/phrases_financieres.csv", index=False)

    # Analyse
    df_result = analyser(df, mode=args.mode, nlp=nlp)
    df_result.to_csv("data/resultats_sentiment.csv", index=False)

    # Exemples
    print("\n[EXEMPLES] Predictions :")
    for _, row in df_result.head(6).iterrows():
        correct = "OK" if row["label_predit"] == row["label_reel"] else "!!"
        print("  [{}] {} | pred={} | conf={:.0%} | {}".format(
            correct,
            row["texte"][:65] + "...",
            row["label_predit"].upper(),
            row["confiance"],
            row["ticker"],
        ))

    # Metriques
    calculer_metriques(df_result)

    # Donnees marche reelles (Yahoo Finance)
    print("\n[MARKET] Recuperation des cours boursiers reels (Yahoo Finance)...")
    returns = fetch_cours_boursiers(period="30d")

    # Visualisations principales
    visualiser(df_result, returns=returns)

    # Panneau supplementaire : correlation sentiment / rendement
    if returns is not None:
        visualiser_correlation(df_result, returns)

    print("\n[DONE] Fichiers generes :")
    print("  - data/phrases_financieres.csv")
    print("  - data/resultats_sentiment.csv")
    print("  - sentiment_rapport.png")
    if returns is not None:
        print("  - sentiment_correlation.png  (Yahoo Finance — cours reels)")


if __name__ == "__main__":
    main()
