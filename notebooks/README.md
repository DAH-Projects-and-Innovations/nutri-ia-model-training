# Notebooks — Expérimentations de fine-tuning (Keras)

Contenu importé depuis la branche `fine-tuning` du repo
[`nutri-ia-data-collection`](https://github.com/DAH-Projects-and-Innovations/nutri-ia-data-collection)
(dernier commit `487e27e`, 10 août 2026) — copie de fichiers uniquement, sans l'historique git
ni les images brutes/augmentées d'entraînement.

## Contenu

- `fine_tuning_efficientnet.ipynb`, `fine_tuning_mobilenet_V2.ipynb`, `fine_tuning_resnet.ipynb`
  — fine-tuning **Keras/TensorFlow** (classification softmax, 6 classes) de trois backbones :
  EfficientNetB0, ResNet, MobileNetV2.
- `courbes_entrainement*.png`, `matrice_confusion*.png` — courbes et matrices de confusion
  correspondantes.
- `models/*.keras` — checkpoints entraînés (val_accuracy ≈ 77–86 % selon le run, voir les
  notebooks pour le détail epoch par epoch).

## À savoir avant de réutiliser ces résultats

Ces notebooks utilisent une approche **différente** de celle du repo actuel :
classification directe (Keras, softmax) plutôt qu'apprentissage d'embeddings avec Triplet Loss
(PyTorch/timm, voir [`src/training/finetune_embedding.py`](../src/training/finetune_embedding.py)
et [`src/process/embedding_model.py`](../src/process/embedding_model.py)). Le rapport
`docs/rapport_comparaison_modeles.md` (branche `documentation`) semble avoir été rédigé en
s'appuyant sur un modèle fine-tuné dans le repo `nutri-ia-data-collection`, pas sur le modèle
`models/embedding_model_finetuned.pt` produit par ce repo — d'où l'écart avec
`reports/comparison_report.txt`, qui lui mesure bien le modèle local.

Pour exécuter ces notebooks ici, il manque les dépendances TensorFlow/Keras
(`tensorflow`, `jupyter`, `seaborn` — non listées dans `pyproject.toml` de ce repo).
