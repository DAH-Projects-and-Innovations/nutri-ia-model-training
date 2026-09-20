# Rapport de comparaison — Meilleur modèle de fine-tuning (Keras) vs modèle d'embedding (PyTorch)

**Date :** 20 septembre 2026
**Statut :** Finalisé

---

## 1. Contexte

Deux approches indépendantes ont été explorées pour la reconnaissance visuelle de plats
ouest-africains (6 classes : Alloco, Foutou, Kedjenou, Mafé, Thiéboudiène, Yassa poulet) :

| Approche | Description |
|---|---|
| **Notebooks Keras** (`notebooks/`) | Classification directe (softmax) de trois backbones (EfficientNetB0, ResNet50, MobileNetV2), fine-tunés en 2 phases. |
| **Modèle d'embedding** (`src/process/embedding_model.py`) | EfficientNet-B2 (backbone gelé) + tête de projection 256D, entraînée par Triplet Loss pour la recherche par similarité. |

Ce rapport compare le **meilleur des trois modèles Keras (ResNet50)** au **modèle
d'embedding fine-tuné**, tous deux évalués sur un vrai jeu de test jamais vu en
entraînement.

> Un précédent rapport de comparaison (branche `documentation`, PR #4) contenait des
> chiffres incohérents avec les mesures réelles et n'a jamais été mergé — ce document le
> remplace.

## 2. Un point important avant les chiffres : deux tâches différentes

Ces deux modèles ne répondent **pas à la même question** :

- **ResNet50 (classification)** répond à *"parmi ces 6 plats, lequel est-ce ?"* — un ensemble
  fermé et fixe de classes. Ajouter un 7ᵉ plat demande de ré-entraîner (au moins) la tête de
  classification.
- **Modèle d'embedding (recherche par similarité)** répond à *"quelle(s) photo(s) déjà connue(s)
  ressemble(nt) le plus à celle-ci ?"* — ajouter un nouveau plat ne demande qu'à indexer de
  nouvelles photos dans `embeddings.h5`, sans ré-entraînement. C'est ce qui permettrait un
  catalogue extensible.

Le tableau ci-dessous compare donc des métriques natives à chaque tâche, pas une métrique
unique. Les deux évaluations utilisent cependant un protocole équivalent : **un vrai test set
jamais vu ni en entraînement ni en sélection du meilleur checkpoint**, avec un split
train/val/test ≈ 70/15/15 dans les deux cas — pour que le chiffre "% de bonnes réponses sur des
photos inédites" soit comparable en pratique, même si le mécanisme diffère.

## 3. Protocoles d'évaluation

| | ResNet50 (Keras) | Modèle d'embedding (PyTorch) |
|---|---|---|
| Split | 70 % train / 15 % val / 15 % test, groupé par photo source (les variantes augmentées d'une même photo ne sont jamais dispersées entre les splits) | Identique (voir `src/training/finetune_embedding.py`) |
| Images de test | 988 | 899 (requêtes) — cherchées contre une galerie de 5 986 images (tout le corpus, y compris train/val) |
| Métrique | Accuracy / F1 (classification directe) | Recall@K (le bon plat est-il parmi les K plus proches voisins ?) |
| Sélection du meilleur checkpoint | Meilleure `val_loss` (early stopping) | Meilleure `val_loss` (Triplet Loss) |
| Dataset | `nutri-ia-data-collection/data/raw/augmented_images` (6 586 photos sources + variantes augmentées) | Identique |

## 4. Résultats

### ResNet50 (Keras) — test set, 988 images

```
              precision    recall  f1-score   support

      alloco       0.99      1.00      1.00       165
      foutou       0.95      0.99      0.97       165
    kedjenou       0.99      0.96      0.98       165
        mafe       0.96      0.96      0.96       163
thieboudiene       0.98      1.00      0.99       165
yassa-poulet       1.00      0.96      0.98       165

    accuracy                           0.98       988
   macro avg       0.98      0.98      0.98       988
```

### Modèle d'embedding (PyTorch) — 899 requêtes de test, galerie de 5 986 images

| Métrique | Score |
|---|---|
| Recall@1 | 81,2 % |
| Recall@3 | 86,9 % |
| Recall@5 | 88,8 % |
| Recall@10 | 91,3 % |

Recall@5 par classe :

| Classe | Recall@5 |
|---|---|
| Alloco | 98,6 % |
| Foutou | 96,3 % |
| Kedjenou | 75,7 % |
| Mafé | 81,4 % |
| Thiéboudiene | 94,4 % |
| Yassa poulet | 82,1 % |

### Comparaison directe

| | ResNet50 (classification) | Embedding (Recall@1) |
|---|---|---|
| "Bonne réponse dès le 1er essai" | **98 %** (accuracy) | 81,2 % (Recall@1) |

Le Recall@1 (le plus proche voisin est-il du bon plat) est la métrique la plus proche d'une
accuracy de classification — c'est la base de comparaison la plus directe entre les deux
approches.

## 5. Analyse

- **ResNet50 surclasse nettement le modèle d'embedding** sur cette mesure (98 % vs 81,2 %),
  un écart de 17 points qui n'est pas dû à un biais de mesure — les deux évaluations utilisent
  un test set réellement jamais vu.
- Les classes les plus faibles pour l'embedding (Kedjenou 75,7 %, Mafé 81,4 %, Yassa poulet
  82,1 %) sont aussi celles identifiées comme les plus ambiguës visuellement dans l'analyse des
  notebooks Keras (confusions Kedjenou ↔ Foutou/Mafé) — cohérent d'une approche à l'autre.
- Deux facteurs plausibles pour expliquer l'écart :
  1. **Backbone différent** : ResNet50 (23M paramètres, entièrement dégelé en Phase 2) vs
     EfficientNet-B2 (7,7M paramètres, backbone **resté gelé** — seule la tête de projection
     256D est entraînée). Le modèle d'embedding a beaucoup moins de capacité adaptée au domaine.
  2. **Objectif d'entraînement** : la Triplet Loss avec sélection aléatoire de triplets (pas de
     hard-negative mining) donne un signal d'apprentissage plus faible que la classification
     softmax directe.

## 6. Recommandation

Pour la tâche actuelle (reconnaissance parmi 6 plats fixes), **ResNet50 (Keras) est
aujourd'hui le modèle le plus précis** et le choix recommandé si l'objectif est uniquement la
classification sur cet ensemble fermé.

Le modèle d'embedding reste pertinent si l'objectif produit est un **catalogue extensible**
(ajouter des plats sans ré-entraîner un classifieur) — mais ses performances actuelles (81,2 %
Recall@1) doivent être améliorées avant un déploiement : dégeler partiellement le backbone
(`unfreeze_last_n_blocks`, actuellement non utilisé et à corriger — voir issues GitHub) et/ou
mettre en place un hard-negative mining pour la Triplet Loss sont les pistes les plus probables.

## 7. Reproduire ces résultats

```bash
# Modèle d'embedding
uv run python src/training/finetune_embedding.py --data-dir <dossier images>
uv run python main.py embed --input <même dossier> --checkpoint models/embedding_model_finetuned.pt --output data/processed/embeddings
uv run python scripts/evaluate/compute_recall.py

# Modèles Keras — voir notebooks/README.md
```
