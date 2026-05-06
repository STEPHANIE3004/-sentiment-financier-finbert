"""
sentiment_finbert.py - Analyse de sentiment financier avec FinBERT
Auteure : Vanelle Stephanie MANGOUA DJOUSSEU

Architecture : ProsusAI/finbert (BERT fine-tune sur textes financiers)
Cas d'usage  : analyse de news financieres, rapports annuels, tweets bourse

Note : ce script fonctionne en 2 modes :
  - Mode DEMO (defaut) : pipeline complet avec donnees synthetiques,
    sans telechargement du modele (utile si pas de GPU / connexion limitee)
  - Mode FINBERT : utilise le vrai modele HuggingFace (necessite ~500MB)
    python sentiment_finbert.py --mode finbert
"""

import argparse
import json
import os
import time
import random
from datetime import datetime, timedelta
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use("Agg")  # desactive pour affichage fenetre
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
        row["ticker"] = random.choice(["BNP.PA", "SG.PA", "ACA.PA",
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
# Visualisations
# --------------------------------------------------------------------------

def visualiser(df):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Analyse Sentiment Financier - FinBERT Pipeline",
                  fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

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

    # (4) Evolution temporelle du sentiment
    ax4 = fig.add_subplot(gs[1, 0:2])
    df_time = df.groupby(["date", "label_predit"]).size().unstack(fill_value=0)
    for lbl in labels:
        if lbl in df_time.columns:
            ax4.plot(df_time.index, df_time[lbl], marker="o",
                     label=lbl, color=COLORS[lbl], linewidth=2)
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Nombre de mentions")
    ax4.set_title("Evolution temporelle du sentiment")
    ax4.legend()
    ax4.tick_params(axis="x", rotation=30)
    ax4.grid(alpha=0.3)

    # (5) Sentiment par ticker
    ax5 = fig.add_subplot(gs[1, 2])
    pivot = df.groupby(["ticker", "label_predit"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=labels, fill_value=0)
    bottom = np.zeros(len(pivot))
    for lbl, col in COLORS.items():
        if lbl in pivot.columns:
            vals = pivot[lbl].values
            ax5.barh(pivot.index, vals, left=bottom, label=lbl, color=col)
            bottom += vals
    ax5.set_xlabel("Nombre de mentions")
    ax5.set_title("Sentiment par valeur boursiere")
    ax5.legend(fontsize=8)
    ax5.grid(axis="x", alpha=0.3)

    out = "sentiment_rapport.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("[VIZ] Rapport sauvegarde : {}".format(out))


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

    # Visualisations
    visualiser(df_result)

    print("\n[DONE] Fichiers generes :")
    print("  - data/phrases_financieres.csv")
    print("  - data/resultats_sentiment.csv")
    print("  - sentiment_rapport.png")


if __name__ == "__main__":
    main()
