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

> Les notebooks d'expérimentation de `notebooks/` (fine-tuning Keras/TensorFlow, voir
> [`notebooks/README.md`](notebooks/README.md)) ont leurs propres dépendances, à installer avec
> `uv sync --extra notebooks`.

## 🧪 Utilisation

### Modèles (.pt)

Les checkpoints du modèle d'embedding (`models/*.pt`) **ne sont pas versionnés** (trop
volumineux pour GitHub) — il faut les générer localement :

```bash
# Modèle de base (backbone gelé, tête de projection non entraînée)
uv run python scripts/save_base_model.py
# → models/embedding_model_base.pt

# Modèle fine-tuné (Triplet Loss sur la tête de projection)
uv run python src/training/finetune_embedding.py --data-dir /chemin/vers/images
# → models/embedding_model_finetuned.pt (sauvegardé au meilleur epoch, sur la loss de validation)
```

`--data-dir` est optionnel (défaut : `data/processed/images` du repo, vide par défaut — les
images viennent de `nutri-ia-data-collection`, voir plus haut).

### Pipeline complet

```bash
# 1. Prétraitement des images brutes (raw → processed)
uv run python main.py prepare --input /chemin/vers/images/brutes

# 2. Génération des embeddings avec un checkpoint donné
uv run python main.py embed --checkpoint models/embedding_model_finetuned.pt

# 3. Évaluation — Recall@K / Precision@K (exclut les variantes augmentées d'une même
#    photo source de la recherche de voisins, voir scripts/evaluate/compute_recall.py)
uv run python scripts/evaluate/compute_recall.py

# 4. Prédiction sur une image isolée (utilise embedding_model_finetuned.pt par défaut,
#    sinon embedding_model_base.pt — voir --checkpoint pour forcer un autre modèle)
uv run python scripts/evaluate/predict_image.py --image chemin/vers/photo.jpg

# 5. Visualisation t-SNE des embeddings
uv run python scripts/evaluate/visualize_embeddings.py
```

### Tests

```bash
uv run --with pytest python -m pytest tests/
```
