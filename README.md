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

## 📊 Résultats — Sélection du modèle (Phase 3)

Le rapport complet de comparaison est disponible dans [`docs/rapport_comparaison_modeles.md`](docs/rapport_comparaison_modeles.md).

**Modèle retenu pour la phase 3 : `models/embedding_model_finetuned.pt`**

| Métrique | Modèle BASE | Modèle FINE-TUNÉ |
|---|---|---|
| Recall@1 | ~60 % | **85,5 %** |
| Recall@5 | ~82 % | **95,9 %** |
| Accuracy KNN-5 | ~57 % | **84,5 %** |
| F1-score macro | ~55 % | **84,4 %** |

Architecture : EfficientNet-B2 (backbone gelé) + tête de projection MLP (256D) entraînée avec
Triplet Loss sur 6 plats ouest-africains (6 586 images).

## 🧪 Utilisation

### Workflow complet

```bash
# 1. Prétraitement des images brutes
uv run python main.py prepare

# 2. Génération des embeddings (modèle fine-tuné)
uv run python main.py embed --checkpoint models/embedding_model_finetuned.pt

# 3. Évaluation — Recall@K
uv run python scripts/evaluate/compute_recall.py

# 4. Comparaison base vs fine-tuné
uv run python scripts/evaluate/compare_models.py

# 5. Visualisation t-SNE
uv run python scripts/evaluate/visualize_embeddings.py

# 6. Prédiction sur une image
uv run python scripts/evaluate/predict_image.py --image chemin/vers/photo.jpg
```

### Fine-tuning

```bash
# Ré-entraîner la tête de projection avec Triplet Loss
uv run python src/training/finetune_embedding.py
```

### Informations sur le modèle actif

```bash
uv run python main.py info
```
