# Nutri-IA : Entraînement, Fine-tuning et Modèles d'Embedding

Ce dépôt s'appuie sur les données produites par [`nutri-ia-data-collection`](https://github.com/DAH-Projects-and-Innovations/nutri-ia-data-collection) et contient le code d'entraînement, de fine-tuning et d'évaluation des modèles (LLM et embeddings) de l'Assistant IA de Nutrition.

## 📁 Architecture du projet

```
nutri-ia-model-training/
├── data
│   ├── raw            <- Copie ou lien vers les données prêtes issues de nutri-ia-data-collection.
│   ├── processed       <- Datasets formatés pour l'entraînement (splits train/val/test).
│   └── embeddings      <- Vecteurs pré-calculés, index (FAISS/Chroma), caches.
├── configs             <- Fichiers de configuration (hyperparamètres, chemins, modèles de base).
├── docs                <- Documentation du projet (choix de modèles, notes méthodologiques, résultats).
├── notebooks           <- Notebooks d'exploration et de prototypage rapide.
├── src
│   ├── data            <- Chargement, tokenization, préparation des datasets.
│   ├── training        <- Scripts de fine-tuning (LLM, classification, etc.).
│   ├── embedding        <- Entraînement/fine-tuning des modèles d'embedding.
│   ├── evaluation       <- Métriques, benchmarks, comparaison de modèles.
│   ├── inference        <- Scripts de test/serving des modèles entraînés.
│   └── utils            <- Fonctions utilitaires communes (logging, seed, config loader).
├── models              <- Checkpoints locaux (non versionnés, sauf configs/cartes de modèle).
├── tests               <- Tests unitaires.
├── main.py             <- Point d'entrée principal.
├── pyproject.toml      <- Gestion des dépendances (via uv).
└── .gitignore          <- Fichiers et dossiers ignorés par Git.
```

## 🔗 Lien avec le projet de collecte de données

Ce dépôt est le second volet du projet Nutri-IA :

- **`nutri-ia-data-collection`** : collecte, nettoyage et préparation des données brutes (images, données tabulaires).
- **`nutri-ia-model-training`** (ce dépôt) : entraînement, fine-tuning et évaluation des modèles à partir des données produites par le premier dépôt.

## 🚀 Installation

Ce projet utilise [uv](https://github.com/astral-sh/uv) pour la gestion rapide des dépendances.

Cloner le dépôt :

```bash
git clone https://github.com/DAH-Projects-and-Innovations/nutri-ia-model-training.git
cd nutri-ia-model-training
```

Installer les dépendances :

```bash
uv sync
```

Activer l'environnement virtuel :

```bash
source .venv/bin/activate
```

## 🧪 Utilisation

_À compléter au fur et à mesure de l'avancement du projet (commandes d'entraînement, d'évaluation, etc.)._
