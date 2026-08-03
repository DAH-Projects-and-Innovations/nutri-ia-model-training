"""
Fine-tuning de la tête de projection avec Triplet Loss.
La tête apprend : même plat → vecteurs proches
                 plats différents → vecteurs éloignés
Usage :
    python src/training/finetune_embedding.py
"""
import sys
from pathlib import Path

# Permet d'importer les modules src depuis la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.process.embedding_model import FoodEmbeddingModel
from src.utils.image_utils import FoodImageDataset, get_train_transforms
from src.config.settings import paths, embed_cfg

# Configuration
EPOCHS     = 20
LR         = 1e-4
BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device utilisé : {DEVICE}")

# Charger le modèle 
# Backbone EfficientNet-B2 gelé — seule la tête de projection sera entraînée
model = FoodEmbeddingModel.from_config(
    freeze_backbone=True,
)
model = model.to(DEVICE)

# Vérification : combien de paramètres sont entraînables ?
counts = model.count_parameters()
print(f"Paramètres total      : {counts['total']:,}")
print(f"Paramètres entraîn.   : {counts['trainable']:,}")
print(f"Paramètres gelés      : {counts['frozen']:,}")

#  Dataset avec augmentation  
# get_train_transforms() applique rotation, zoom, luminosité à la volée
dataset = FoodImageDataset(
    root_dir=paths.processed_images,
    transform=get_train_transforms()
)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,      # 0 sur Windows pour éviter les erreurs de multiprocessing
    drop_last=True,     # ignore le dernier batch incomplet
)

print(f"\nEntraînement sur {len(dataset)} images, {EPOCHS} epochs")
print(f"Batches par epoch : {len(loader)}")
print(f"Classes : {dataset.classes}\n")

# Triplet Loss
# margin=0.3 : force les plats différents à être au moins 0.3 plus loin
loss_fn   = nn.TripletMarginLoss(margin=0.3, p=2)

# On n'optimise QUE la tête de projection, pas le backbone
optimizer = torch.optim.Adam(
    model.projection.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

# Scheduler : réduit le LR si la loss stagne
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5
)

# Entraînement 
best_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss  = 0.0
    n_batches   = 0
    n_triplets_total = 0

    for batch in loader:
        images = batch["image"].to(DEVICE)   # (B, 3, 224, 224)
        labels = batch["label"].to(DEVICE)   # (B,)

        # Calculer les embeddings du batch
        embeddings = model(images)            # (B, 256)

        batch_loss    = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        n_triplets    = 0

        B = len(labels)

        for i in range(B):
            label_i = labels[i]

            # Indices des images du MÊME plat dans ce batch (sauf i lui-même)
            pos_mask = (labels == label_i)
            pos_mask[i] = False
            pos_indices = pos_mask.nonzero(as_tuple=True)[0]

            # Indices des images d'un AUTRE plat dans ce batch
            neg_mask    = (labels != label_i)
            neg_indices = neg_mask.nonzero(as_tuple=True)[0]

            # S'il n'y a pas de positif ou de négatif dans ce batch → passer
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            # Choisir un positif et un négatif au hasard
            # .item() pour obtenir un entier Python — évite les problèmes de dim
            pos_idx = pos_indices[torch.randint(len(pos_indices), (1,)).item()]
            neg_idx = neg_indices[torch.randint(len(neg_indices), (1,)).item()]

            # Extraire les vecteurs — shape (256,) chacun
            anchor   = embeddings[i]       # (256,)
            positive = embeddings[pos_idx] # (256,)
            negative = embeddings[neg_idx] # (256,)

            # Ajouter la dimension batch : (256,) → (1, 256)
            anchor   = anchor.unsqueeze(0)   # (1, 256)
            positive = positive.unsqueeze(0) # (1, 256)
            negative = negative.unsqueeze(0) # (1, 256)

            # Calculer la Triplet Loss pour ce triplet
            t_loss     = loss_fn(anchor, positive, negative)
            batch_loss = batch_loss + t_loss
            n_triplets += 1

        # Si au moins un triplet valide dans ce batch
        if n_triplets > 0:
            batch_loss = batch_loss / n_triplets  # moyenne sur les triplets
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            n_batches  += 1
            n_triplets_total += n_triplets

    # Loss moyenne de l'epoch
    avg_loss = total_loss / max(n_batches, 1)
    scheduler.step(avg_loss)

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Triplets: {n_triplets_total} | "
        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
    )

    # Sauvegarder le meilleur modèle
    if avg_loss < best_loss:
        best_loss = avg_loss
        model.save("models/embedding_model_finetuned.pt")
        print(f"  → Meilleur modèle sauvegardé (loss={best_loss:.4f})")

# Résumé final
print("\n" + "═" * 60)
print("  FINE-TUNING TERMINÉ")
print("═" * 60)
print(f"  Meilleure loss     : {best_loss:.4f}")
print(f"  Modèle sauvegardé  : models/embedding_model_finetuned.pt")
print()
print("  Prochaines étapes :")
print("  1. Régénérer les embeddings avec le nouveau modèle :")
print("     python main.py embed --checkpoint models/embedding_model_finetuned.pt")
print("  2. Réévaluer :")
print("     python scripts/evaluate/compute_recall.py")
print("  3. Visualiser :")
print("     python scripts/evaluate/visualize_embeddings.py")
print("═" * 60)