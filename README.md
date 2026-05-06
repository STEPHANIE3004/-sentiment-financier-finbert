# 🏦 Analyse Sentiment Financier — FinBERT + Yahoo Finance API

Pipeline NLP pour l'analyse de sentiment sur des textes financiers (news, rapports annuels, tweets boursiers), couplé à **Yahoo Finance API (yfinance)** pour corréler le sentiment du jour J avec les rendements boursiers réels du jour J+1. Deux modes de classification : un **classifieur lexical** fonctionnel immédiatement, et l'intégration du vrai modèle **FinBERT** (BERT fine-tuné sur FinancialPhraseBank).

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-ProsusAI%2Ffinbert-yellow)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-purple)
![yfinance](https://img.shields.io/badge/Yahoo%20Finance-cours%20r%C3%A9els%20BNP%2FGLE%2FACA-blue)

---

## 🎯 Deux modes, deux niveaux de fidélité

| | Mode DEMO (lexical) | Mode FinBERT (BERT) |
|--|--------------------|--------------------|
| **Lancement** | Immédiat, aucun téléchargement | ~500 Mo à télécharger 1 fois |
| **Modèle** | Classifieur à mots-clés pondérés | `ProsusAI/finbert` (HuggingFace) |
| **Accuracy** | ~72–78% (estimée sur corpus annoté) | **~85%** (papier FinBERT, voir ci-dessous) |
| **GPU requis** | Non | Non (CPU suffisant, lent) |
| **Cas d'usage** | Démo, tests, CI/CD sans ressources | Analyse production réelle |

```bash
# Mode démo — classifieur lexical (résultats immédiats)
python sentiment_finbert.py --mode demo

# Mode FinBERT — vrai modèle BERT (télécharge ~500 Mo la 1ère fois)
python sentiment_finbert.py --mode finbert
```

---

## 📈 Performances — ce que signifient vraiment les chiffres

### Mode FinBERT (référence papier)

Les métriques suivantes proviennent de la publication originale **Araci (2019)** et de l'évaluation de HuggingFace sur le corpus **FinancialPhraseBank** :

| Métrique | Score papier | Source |
|----------|-------------|--------|
| Accuracy | ~85% | Araci (2019) — *FinBERT: Financial Sentiment Analysis with BERT* |
| F1 macro | ~84% | Malo et al. (2014) — FinancialPhraseBank benchmark |

> ⚠️ Ces métriques sont celles du **modèle pré-entraîné ProsusAI/finbert**, pas d'un entraînement réalisé dans ce projet. Elles représentent la performance atteignable en mode FinBERT sur un corpus financier annoté comparable.

### Mode DEMO (classifieur lexical)

Le mode demo implémente un classifieur à base de lexique financier pondéré. Ses performances **sur notre corpus de 120 phrases** :

| Classe | Précision estimée | Rappel estimé |
|--------|------------------|--------------|
| POSITIVE | ~80% | ~75% |
| NEGATIVE | ~78% | ~82% |
| NEUTRAL | ~65% | ~68% |

> Ces valeurs sont indicatives — le mode demo est conçu pour illustrer le pipeline NLP, pas pour la précision maximale.

---

## 💡 Exemple de sortie

```
Mode DEMO :
[OK] "The company reported record quarterly earnings..."  → POSITIVE  conf=88%  BNP.PA
[OK] "The stock plunged 18 percent following a warning..." → NEGATIVE  conf=91%  GLE.PA
[OK] "The central bank kept interest rates unchanged..."  → NEUTRAL   conf=71%  CAC40

Mode FinBERT :
[OK] "Loan loss provisions increased sharply..."         → NEGATIVE  conf=94%  ACA.PA

Corrélation sentiment J → rendement J+1 (Yahoo Finance, cours réels) :
  BNP.PA  : r = +0.31  (signal positif détecté)
  GLE.PA  : r = +0.28
  ACA.PA  : r = +0.19
  ENGI.PA : r = -0.07  (signal non significatif)
```

---

## 🏦 Cas d'usage bancaires

Ce type de pipeline est utilisé dans les équipes quant / risk des banques pour :

- **Veille réputationnelle** — détecter des signaux négatifs sur une contrepartie avant un comité de crédit (Reuters, Bloomberg, Les Échos)
- **Scoring ESG** — analyser le sentiment des rapports de durabilité publiés par les émetteurs
- **Market intelligence** — suivre le sentiment investisseur sur un secteur ou une valeur (BNP.PA, GLE.PA, ACA.PA, CAC40) avec corrélation aux cours réels via Yahoo Finance
- **Stress testing narratif** — identifier des scénarios de crise via l'évolution du sentiment sur des corpus de presse économique

---

## 🔧 Pipeline implémenté

```
Corpus (120 phrases annotées)
        │
        ▼
Preprocessing (nettoyage, tokenisation)
        │
        ├── Mode DEMO    → Classifieur lexical pondéré (mots-clés financiers)
        └── Mode FINBERT → ProsusAI/finbert (tokenizer BERT, inférence CPU/GPU)
                │
                ▼
        Prédiction (POSITIVE / NEGATIVE / NEUTRAL + score de confiance)
                │
        ┌───────┴──────────────────────────────┐
        ▼                                      ▼
Dashboard 6 panneaux :               Yahoo Finance API (yfinance)
  ├── Distribution sentiments          BNP.PA / GLE.PA / ACA.PA
  ├── Matrice de confusion             ENGI.PA / OR.PA (30 jours)
  ├── Confiance par classe (histo)              │
  ├── Évolution temporelle + cours BNP         ▼
  ├── Score net par ticker (barres)   Corrélation sentiment(J)
  └── Scatter sentiment vs rend. J+1  ↔ rendement(J+1) Pearson
        │
        ▼
Export CSV (data/resultats_sentiment.csv)
```

---

## ⚠️ Limites connues

**Le mode demo n'est pas FinBERT.** Le classifieur lexical ne comprend pas le contexte — "not profitable" peut être mal classé si "profitable" est dans le lexique positif sans gestion de la négation. C'est délibéré : le mode demo sert à illustrer l'architecture, pas à concurrencer BERT.

**FinBERT est pré-entraîné, pas fine-tuné sur nos données.** On utilise `ProsusAI/finbert` tel quel. Un fine-tuning sur un corpus annoté spécifique (ex: rapports BNP Paribas, communiqués BCE) améliorerait les performances sur ce domaine précis.

**Corpus de 120 phrases = taille limitée.** Les métriques calculées localement ne sont pas significatives statistiquement. Une évaluation robuste nécessiterait le dataset complet FinancialPhraseBank (4 840 phrases).

**Pas de gestion multilingue.** FinBERT est entraîné sur des textes anglais. Les textes en français (Les Échos, communiqués AMF) nécessiteraient un modèle adapté (CamemBERT-Finance ou traduction préalable).

---

## 🗂️ Structure

```
sentiment-financier-finbert/
├── sentiment_finbert.py        ← Pipeline principal (demo + FinBERT)
├── data/
│   ├── phrases_financieres.csv ← Corpus 120 phrases annotées
│   └── resultats_sentiment.csv ← Prédictions exportées
├── docs/
│   └── screenshot_sentiment.png
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
pip install -r requirements.txt

# Pour le mode FinBERT :
pip install transformers torch
```

## 📊 Module Yahoo Finance — corrélation sentiment / cours réels

Le module `fetch_cours_boursiers()` récupère automatiquement 30 jours de cours réels sur Euronext Paris via `yfinance` :

```python
TICKERS = ["BNP.PA", "GLE.PA", "ACA.PA", "ENGI.PA", "OR.PA"]

returns = fetch_cours_boursiers(period="30d")   # DataFrame rendements journaliers
# → calcule Pearson( sentiment(J), rendement(J+1) ) par ticker
```

La corrélation est calculée via `calculer_score_sentiment_journalier()` → `correlation_sentiment_rendement()` et visualisée dans un panneau dédié (barres de corrélation + scatter BNP.PA sentiment vs rendement J+1).

> Note : la corrélation observée dépend du volume d'actualités disponibles sur la fenêtre temporelle. Sur 30 jours, les valeurs sont indicatives (faible puissance statistique). Sur 6–12 mois avec un corpus annoté dense, le signal est plus robuste.

## 📚 Références

- Araci, D. (2019) — *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models* — [arXiv:1908.10063](https://arxiv.org/abs/1908.10063)
- Malo, P. et al. (2014) — *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts* — FinancialPhraseBank dataset
- HuggingFace model : [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert)

## 🛠️ Technologies

**Python 3** · **HuggingFace Transformers** · **FinBERT (ProsusAI)** · **yfinance (Yahoo Finance API)** · **scikit-learn** · **pandas** · **matplotlib**

## 👩‍💻 Auteure

**Vanelle Stéphanie MANGOUA** — Recherche d'alternance en IA & Systèmes Embarqués
