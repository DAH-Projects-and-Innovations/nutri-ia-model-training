"""
Fine-tuning de la tête de projection avec Triplet Loss.
La tête apprend : même plat → vecteurs proches
                 plats différents → vecteurs éloignés
Usage :
    python src/training/finetune_embedding.py
    python src/training/finetune_embedding.py --data-dir /chemin/vers/images
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# Permet d'importer les modules src depuis la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.process.embedding_model import FoodEmbeddingModel
from src.utils.image_utils import FoodImageDataset, get_inference_transforms, get_train_transforms
from src.config.settings import paths, embed_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning de la tête de projection avec Triplet Loss."
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help=(
            "Répertoire d'images pour l'entraînement, sous-dossiers = classes "
            "(défaut : data/processed/images du repo)."
        ),
    )
    return parser.parse_args()


args = parse_args()

# Configuration
DATA_DIR    = Path(args.data_dir) if args.data_dir else paths.processed_images
EPOCHS      = 20
LR          = 1e-4
BATCH_SIZE  = 32
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15  # jamais vu en entraînement NI en validation — même ratio que les
                    # notebooks Keras (70/15/15), pour une comparaison base/fine-tuné
                    # honnête entre les deux approches.
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_PATHS_FILE = Path("reports/embedding_test_paths.txt")

print(f"Device utilisé : {DEVICE}")
print(f"Dossier d'images : {DATA_DIR}")

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

def source_group_id(path: Path) -> str:
    """
    Identifiant de la photo source à partir du nom de fichier, pour grouper les
    variantes augmentées d'une même photo (ex. ``foutou_001_pil_env_...jpg`` et
    ``foutou_001_sd_calebasse.jpg`` → même groupe ``"foutou_001"``) et éviter
    qu'elles se retrouvent dispersées entre train et val : sans ça, la loss de
    validation voit des quasi-doublons de photos déjà vues en entraînement, et
    le Recall@K mesuré ensuite sur ce type de split est artificiellement gonflé
    (le plus proche voisin d'une variante est souvent une autre variante de la
    même photo, pas une vraie généralisation).

    Si le nom de fichier ne contient pas de suffixe numérique de ce type,
    chaque fichier reste son propre groupe (comportement inchangé — c'est le
    cas du dataset non augmenté, une photo = un fichier).
    """
    match = re.match(r"^(.+?_\d+)", path.stem)
    return match.group(1) if match else path.stem


#  Dataset avec split train/val, groupé par photo source
# Deux instances du même dossier : train_tf pour l'augmentation à la volée,
# val_tf (déterministe, sans augmentation) pour que la loss de validation soit
# stable d'une epoch à l'autre. Le même tirage d'indices est utilisé pour les
# deux, donc train/val portent bien sur les mêmes images des deux côtés.
dataset_train_view = FoodImageDataset(
    root_dir=DATA_DIR,
    transform=get_train_transforms(),
)
dataset_val_view = FoodImageDataset(
    root_dir=DATA_DIR,
    transform=get_inference_transforms(),
)

groups: dict[str, list[int]] = defaultdict(list)
for idx, img_path in enumerate(dataset_train_view.image_paths):
    groups[source_group_id(img_path)].append(idx)

group_ids = list(groups.keys())
n_total = len(dataset_train_view)
n_test_target = int(n_total * TEST_SPLIT)
n_val_target  = int(n_total * VAL_SPLIT)

shuffled_groups = torch.randperm(
    len(group_ids), generator=torch.Generator().manual_seed(SEED)
).tolist()

# Test réservé en premier, jamais touché par l'entraînement ni la sélection du
# meilleur checkpoint (val) — sert uniquement à l'évaluation finale, pour une
# comparaison à armes égales avec le protocole des notebooks Keras.
test_indices: list[int]  = []
val_indices: list[int]   = []
train_indices: list[int] = []
for i in shuffled_groups:
    if len(test_indices) < n_test_target:
        bucket = test_indices
    elif len(val_indices) < n_val_target:
        bucket = val_indices
    else:
        bucket = train_indices
    bucket.extend(groups[group_ids[i]])

n_train = len(train_indices)
n_val   = len(val_indices)
n_test  = len(test_indices)

train_dataset = Subset(dataset_train_view, train_indices)
val_dataset   = Subset(dataset_val_view, val_indices)
# Pas de test_dataset/loader ici : le test set n'est jamais passé au modèle dans
# ce script, seulement réservé. L'évaluation finale se fait après-coup par
# compute_recall.py, sur les embeddings régénérés pour tout le corpus.

# Sauvegarde des chemins du test set — nécessaire pour restreindre l'évaluation
# finale (compute_recall.py) à ces images jamais vues, plutôt que tout le corpus.
TEST_PATHS_FILE.parent.mkdir(parents=True, exist_ok=True)
test_paths = [str(dataset_train_view.image_paths[i]) for i in test_indices]
with open(TEST_PATHS_FILE, "w") as f:
    f.write("\n".join(test_paths))
print(f"Split de test sauvegardé → {TEST_PATHS_FILE} ({n_test} images, jamais vues en train/val)")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,      # 0 sur Windows pour éviter les erreurs de multiprocessing
    drop_last=True,     # ignore le dernier batch incomplet
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)

print(f"\nEntraînement sur {n_train} images / validation sur {n_val} images / test {n_test} images (réservé), {EPOCHS} epochs")
print(f"Batches par epoch (train) : {len(train_loader)}")
print(f"Classes : {dataset_train_view.classes}\n")

# Triplet Loss
# margin=0.3 : force les plats différents à être au moins 0.3 plus loin
loss_fn   = nn.TripletMarginLoss(margin=0.3, p=2)

# On n'optimise QUE la tête de projection, pas le backbone
optimizer = torch.optim.Adam(
    model.projection.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

# Scheduler : réduit le LR si la loss de validation stagne
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5
)


def batch_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor | None, int]:
    """
    Construit les triplets (ancre, positif, négatif) disponibles dans un batch
    et retourne la Triplet Loss moyenne du batch, ainsi que le nombre de
    triplets utilisés. Retourne ``(None, 0)`` si aucun triplet n'est possible
    (pas de positif ou de négatif disponible pour aucune image du batch).
    """
    batch_loss = torch.zeros((), device=embeddings.device)
    n_triplets = 0
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

    if n_triplets == 0:
        return None, 0
    return batch_loss / n_triplets, n_triplets


# Entraînement
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    # --- Phase entraînement ---
    # model.train() garde automatiquement le backbone gelé en eval() —
    # voir FoodEmbeddingModel.train() dans src/process/embedding_model.py.
    model.train()
    total_loss = 0.0
    n_batches  = 0
    n_triplets_total = 0

    for batch in train_loader:
        images = batch["image"].to(DEVICE)   # (B, 3, 224, 224)
        labels = batch["label"].to(DEVICE)   # (B,)

        # Calculer les embeddings du batch
        embeddings = model(images)            # (B, 256)
        batch_loss, n_triplets = batch_triplet_loss(embeddings, labels)

        # Si au moins un triplet valide dans ce batch
        if batch_loss is not None:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            n_batches  += 1
            n_triplets_total += n_triplets

    avg_train_loss = total_loss / max(n_batches, 1)

    # --- Phase validation (pas de gradient, pas de mise à jour des poids) ---
    model.eval()
    val_loss_total = 0.0
    val_batches    = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            embeddings = model(images)
            batch_loss, _ = batch_triplet_loss(embeddings, labels)
            if batch_loss is not None:
                val_loss_total += batch_loss.item()
                val_batches += 1

    avg_val_loss = val_loss_total / max(val_batches, 1)
    scheduler.step(avg_val_loss)

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Train loss: {avg_train_loss:.4f} | "
        f"Val loss: {avg_val_loss:.4f} | "
        f"Triplets (train): {n_triplets_total} | "
        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
    )

    # Sauvegarder le meilleur modèle — sur la loss de VALIDATION, pas la loss
    # d'entraînement (qui ne protège pas du surapprentissage).
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save("models/embedding_model_finetuned.pt")
        print(f"  → Meilleur modèle sauvegardé (val_loss={best_val_loss:.4f})")

# Résumé final
print("\n" + "═" * 60)
print("  FINE-TUNING TERMINÉ")
print("═" * 60)
print(f"  Meilleure val_loss : {best_val_loss:.4f}")
print(f"  Modèle sauvegardé  : models/embedding_model_finetuned.pt")
print(f"  Split de test      : {TEST_PATHS_FILE} ({n_test} images, jamais entraînées)")
print()
print("  Prochaines étapes :")
print("  1. Régénérer les embeddings (tout le corpus, sert de galerie de recherche) :")
print("     python main.py embed --checkpoint models/embedding_model_finetuned.pt")
print("  2. Réévaluer — restreint automatiquement aux images de test si")
print(f"     {TEST_PATHS_FILE} existe :")
print("     python scripts/evaluate/compute_recall.py")
print("  3. Visualiser :")
print("     python scripts/evaluate/visualize_embeddings.py")
print("═" * 60)
