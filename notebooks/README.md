# Notebooks — Expérimentations de fine-tuning (Keras)

Contenu importé depuis la branche `fine-tuning` du repo
[`nutri-ia-data-collection`](https://github.com/DAH-Projects-and-Innovations/nutri-ia-data-collection)
(dernier commit `487e27e`, 10 août 2026) — copie de fichiers uniquement, sans l'historique git
ni les images brutes/augmentées d'entraînement.

## Contenu

- `fine_tuning_efficientnet.ipynb`, `fine_tuning_mobilenet_V2.ipynb`, `fine_tuning_resnet.ipynb`
  — fine-tuning **Keras/TensorFlow** (classification softmax, 6 classes) de trois backbones :
  EfficientNetB0, ResNet50, MobileNetV2.
- `courbes_entrainement*.png`, `matrice_confusion*.png` — courbes et matrices de confusion
  correspondantes.
- `models/*_phase1.keras` — checkpoint après la Phase 1 (backbone gelé, seule la tête est
  entraînée). `models/*_finetuned.keras` — checkpoint après la Phase 2 (dernières couches du
  backbone dégelées, LR très faible) : c'est celui-ci qui fait foi comme résultat final.
- `models/*_classes.json` — ordre des classes correspondant aux index de sortie de chaque
  modèle (nécessaire pour interpréter les prédictions).
- `../reports/keras_*_report.txt` — rapport de classification (test set) pour chaque
  architecture, dans le même esprit que `reports/recall_report.txt` côté embeddings.

Chaque notebook fait maintenant un vrai split **train/val/test (70/15/15) stratifié par
classe**, entraîne en deux temps (Phase 1 : tête seule, backbone gelé — Phase 2 : dégel des
20 dernières couches du backbone à LR=1e-5), et évalue la métrique finale sur le test set
(jamais vu par `EarlyStopping`/`ReduceLROnPlateau`, contrairement à l'ancien split 80/20
train/val qui servait aux deux). Une dernière cellule dans chaque notebook montre comment
recharger le modèle et prédire sur une image isolée.

Les 3 notebooks pointent vers le même dossier d'images
(`../../nutri-ia-data-collection/data/raw/images`) pour rester comparables entre eux —
auparavant EfficientNet/MobileNetV2 et ResNet utilisaient des chemins différents.

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
