"""
Script 4 — Prédiction sur une nouvelle image
═══════════════════════════════════════════════════
Donne une image au modèle et il te dit quel plat c'est.
C'est le test le plus concret pour évaluer les performances.

Usage :
    python scripts/evaluate/predict_image.py --image chemin/vers/photo.jpg
    python scripts/evaluate/predict_image.py --image data/test/photo_thieb.jpg
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

# Chemin vers le modèle et les embeddings
EMBEDDINGS_H5  = Path("data/processed/embeddings/embeddings.h5")
MODEL_PT       = Path("models/embedding_model.pt")

NOM_CLASSES = {
    0: "Alloco",
    1: "Foutou",
    2: "Kedjenou",
    3: "Mafé",
    4: "Thiéboudiene",
    5: "Yassa poulet",
}

def predire(image_path: str):
    from src.process.embedding_model import FoodEmbeddingModel
    from src.utils.image_utils import get_inference_transforms

    print("═" * 60)
    print("  PRÉDICTION SUR UNE NOUVELLE IMAGE")
    print("═" * 60)
    print(f"\n  Image : {image_path}\n")

    #  Étape 1 : Charger le modèle 
    print("[1] Chargement du modèle...")
    if MODEL_PT.exists():
        model = FoodEmbeddingModel.load(MODEL_PT)
        print(f"  Modèle chargé depuis {MODEL_PT}")
    else:
        from src.config.settings import embed_cfg
        model = FoodEmbeddingModel.from_config(freeze_backbone=True)
        print("  Modèle créé depuis la config (pas de checkpoint)")
    model.eval()

    #  Étape 2 : Préparer l'image 
    print("\n[2] Préparation de l'image...")
    transform = get_inference_transforms()

    try:
        img = Image.open(image_path).convert("RGB")
        print(f"  Taille originale : {img.size[0]}×{img.size[1]} px")
    except Exception as e:
        print(f"  ✗   Impossible d'ouvrir l'image : {e}")
        return

    # Ajouter une dimension batch : (3, 224, 224) → (1, 3, 224, 224)
    tensor = transform(img).unsqueeze(0)

    #  Étape 3 : Calculer l'embedding de la nouvelle image 
    print("\n[3] Calcul de l'embedding...")
    with torch.no_grad():
        embedding_nouveau = model.encode(tensor, return_numpy=True)  # (1, 256)

    print(f"  Vecteur calculé : 256 dimensions")
    print(f"  Norme L2        : {np.linalg.norm(embedding_nouveau):.6f}  (doit être ≈ 1.0)")

    #  Étape 4 : Chercher les plus proches dans embeddings.h5 
    print("\n[4] Recherche des images similaires dans la base...")
    with h5py.File(EMBEDDINGS_H5, "r") as f:
        embeddings_base = f["embeddings"][:]   # (N, 256)
        labels_base     = f["labels"][:]       # (N,)
        paths_base      = f["paths"][:]        # (N,)

    # Similarité cosinus = produit scalaire (vecteurs normalisés L2)
    # embedding_nouveau : (1, 256)
    # embeddings_base   : (N, 256)
    # sims              : (N,)
    sims = (embeddings_base @ embedding_nouveau.T).squeeze()

    # Top 5 plus proches
    top5_idx = np.argsort(sims)[::-1][:5]

    #  Étape 5 : Afficher les résultats 
    print("\n[5] Résultats\n")
    print("─" * 60)

    # Prédiction principale = le voisin le plus proche
    idx_meilleur    = top5_idx[0]
    label_predit    = int(labels_base[idx_meilleur])
    nom_predit      = NOM_CLASSES.get(label_predit, f"classe_{label_predit}")
    score_confiance = sims[idx_meilleur]

    print(f"  🍽️  PLAT DÉTECTÉ : {nom_predit}")
    print(f"  📊  Confiance    : {score_confiance:.1%}")
    print()
    print("  Top 5 images les plus similaires :")
    print("─" * 60)

    for rang, idx in enumerate(top5_idx, 1):
        label  = int(labels_base[idx])
        nom    = NOM_CLASSES.get(label, f"classe_{label}")
        chemin = paths_base[idx].decode()
        sim    = sims[idx]
        barre  = "█" * int(sim * 20)
        print(f"  {rang}. {nom:20s}  sim={sim:.4f}  {barre}")
        print(f"     {chemin}")
        print()

    #  Étape 6 : Verdict 
    print("─" * 60)

    # Vérifier si tous les top5 sont du même plat (bonne cohérence)
    labels_top5    = [int(labels_base[i]) for i in top5_idx]
    tous_pareil    = len(set(labels_top5)) == 1
    majorite_label = max(set(labels_top5), key=labels_top5.count)
    nom_majorite   = NOM_CLASSES.get(majorite_label, "?")
    nb_majorite    = labels_top5.count(majorite_label)

    print(f"\n  VERDICT :")
    if tous_pareil:
        print(f"  ✓   Les 5 voisins sont tous '{nom_predit}'")
        print(f"     Le modèle est très confiant.")
    elif nb_majorite >= 3:
        print(f"  ✓   {nb_majorite}/5 voisins sont '{nom_majorite}'")
        print(f"     Le modèle est confiant.")
    else:
        print(f"  ▲  Les voisins sont partagés entre plusieurs plats.")
        print(f"     Le modèle hésite — image ambiguë ou plat peu représenté.")

    print("═" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prédit le plat sur une nouvelle image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Chemin vers l'image à tester (JPG ou PNG)"
    )
    args = parser.parse_args()
    predire(args.image)