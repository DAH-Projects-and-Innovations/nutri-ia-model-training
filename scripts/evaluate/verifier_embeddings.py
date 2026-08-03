"""
Script 1 — Vérification du fichier embeddings.h5
═══════════════════════════════════════════════════
Vérifie que le fichier produit par main.py embed est correct.

Usage :
    python scripts/evaluate/verifier_embeddings.py

Ce script ne modifie rien — il lit et affiche seulement.
"""

import sys
from pathlib import Path

import h5py
import numpy as np

# Chemin du fichier à vérifier 
EMBEDDINGS_H5 = Path("data/processed/embeddings/embeddings.h5")

# Noms des classes 
NOM_CLASSES = {
    0: "alloco",
    1: "foutou",
    2: "kedjenou",
    3: "mafé",
    4: "thiéboudiene",
    5: "yassa-poulet",
}


def ligne(car="─", n=60):
    print(car * n)

def ok(msg):
    print(f"  ✓   {msg}")

def warn(msg):
    print(f"  ▲   {msg}")

def erreur(msg):
    print(f"  ✗   {msg}")


def verifier():
    ligne("═")
    print("  VÉRIFICATION DU FICHIER EMBEDDINGS")
    ligne("═")

    # CHECK 0 : le fichier existe 
    print("\n[1] Existence du fichier")
    ligne()
    if not EMBEDDINGS_H5.exists():
        erreur(f"Fichier introuvable : {EMBEDDINGS_H5}")
        erreur("Lance d'abord :  uv run python main.py embed")
        sys.exit(1)
    taille_mb = EMBEDDINGS_H5.stat().st_size / 1024 / 1024
    ok(f"Fichier trouvé : {EMBEDDINGS_H5}")
    ok(f"Taille sur disque : {taille_mb:.1f} Mo")

    # CHECK 1 : lecture et structure 
    print("\n[2] Structure interne du fichier HDF5")
    ligne()
    with h5py.File(EMBEDDINGS_H5, "r") as f:
        datasets = list(f.keys())
        print(f"  Datasets présents : {datasets}")

        if "embeddings" not in datasets:
            erreur("Dataset 'embeddings' manquant — fichier corrompu")
            sys.exit(1)
        if "paths" not in datasets:
            warn("Dataset 'paths' manquant — normal si format npy utilisé")
        if "labels" not in datasets:
            warn("Dataset 'labels' manquant — les classes ne seront pas vérifiées")

        embeddings = f["embeddings"][:]
        paths      = f["paths"][:] if "paths" in datasets else None
        labels     = f["labels"][:] if "labels" in datasets else None

        # Métadonnées
        if "embedding_dim" in f.attrs:
            ok(f"embedding_dim (métadonnée) : {f.attrs['embedding_dim']}")
        if "n_samples" in f.attrs:
            ok(f"n_samples (métadonnée)     : {f.attrs['n_samples']}")

    # CHECK 2 : shape 
    print("\n[3] Shape des embeddings")
    ligne()
    N, D = embeddings.shape
    print(f"  Shape : ({N}, {D})")

    if D == 256:
        ok(f"Dimension correcte : {D}D (attendu 256)")
    else:
        warn(f"Dimension inattendue : {D}D (attendu 256) — vérifie settings.py")

    if N == 0:
        erreur("Aucun embedding ! Le fichier est vide.")
        sys.exit(1)
    else:
        ok(f"Nombre total d'images : {N}")

    if embeddings.dtype == np.float32:
        ok(f"Type correct : float32")
    else:
        warn(f"Type : {embeddings.dtype} (attendu float32)")

    # CHECK 3 : normalisation L2 
    print("\n[4] Normalisation L2 (norme de chaque vecteur)")
    ligne()
    normes = np.linalg.norm(embeddings, axis=1)
    norme_min  = normes.min()
    norme_max  = normes.max()
    norme_moy  = normes.mean()
    norme_std  = normes.std()

    print(f"  Norme min  : {norme_min:.6f}")
    print(f"  Norme max  : {norme_max:.6f}")
    print(f"  Norme moy  : {norme_moy:.6f}")
    print(f"  Norme std  : {norme_std:.6f}")

    if norme_max - norme_min < 1e-3:
        ok("Vecteurs normalisés L2 ✓  (toutes les normes ≈ 1.0)")
    else:
        erreur("Vecteurs NON normalisés — vérifie normalize_output=True dans settings.py")

    # CHECK 4 : distribution par classe 
    if labels is not None:
        print("\n[5] Distribution par classe")
        ligne()
        classes_presentes = np.unique(labels)
        print(f"  Nombre de classes : {len(classes_presentes)}")
        print()

        for idx in sorted(classes_presentes):
            count = np.sum(labels == idx)
            nom   = NOM_CLASSES.get(int(idx), f"classe_{idx}")
            barre = "█" * (count // 20)
            statut = "✓" if count >= 50 else "▲  "
            print(f"  {statut}  {nom:20s} : {count:5d} images  {barre}")

        counts = [np.sum(labels == i) for i in classes_presentes]
        if max(counts) / max(min(counts), 1) > 5:
            warn("Dataset déséquilibré — certaines classes ont beaucoup plus d'images")
        else:
            ok("Distribution équilibrée entre les classes")

    # CHECK 5 : exemples de vecteurs 
    print("\n[6] Aperçu de 3 vecteurs")
    ligne()
    for i in [0, N//2, N-1]:
        chemin = paths[i].decode() if paths is not None else f"image_{i}"
        label  = NOM_CLASSES.get(int(labels[i]), "?") if labels is not None else "?"
        vecteur_preview = embeddings[i, :6]
        print(f"  Image {i:5d} | {label:20s} | [{', '.join(f'{v:.3f}' for v in vecteur_preview)}, ...]")

    # CHECK 6 : similarité de sanité 
    if labels is not None and N >= 10:
        print("\n[7] Test de similarité de sanité")
        ligne()
        print("  Prend une image au hasard et cherche ses 3 plus proches voisins.")
        print("  Ils doivent être du même plat.\n")

        idx_query = N // 4
        query     = embeddings[idx_query]
        sims      = embeddings @ query
        sims[idx_query] = -1
        top3      = np.argsort(sims)[::-1][:3]

        nom_query = NOM_CLASSES.get(int(labels[idx_query]), "?")
        chemin_q  = paths[idx_query].decode() if paths is not None else f"image_{idx_query}"
        print(f"  Image requête : {chemin_q}  [{nom_query}]")
        print()

        tous_corrects = True
        for rang, idx in enumerate(top3, 1):
            nom_voisin = NOM_CLASSES.get(int(labels[idx]), "?")
            sim_val    = sims[idx]
            correct    = "✓" if nom_voisin == nom_query else "✗"
            if nom_voisin != nom_query:
                tous_corrects = False
            chemin_v = paths[idx].decode() if paths is not None else f"image_{idx}"
            print(f"  Voisin {rang} : {correct}  {nom_voisin:20s}  sim={sim_val:.4f}  {chemin_v}")

        print()
        if tous_corrects:
            ok("Les 3 voisins sont du même plat — embeddings cohérents ✓")
        else:
            warn("Certains voisins sont d'un autre plat — normal si peu d'images")

    # BILAN FINAL 
    ligne("═")
    print("  BILAN")
    ligne("═")
    print(f"""
  Fichier      : {EMBEDDINGS_H5}
  Images       : {N}
  Dimension    : {D}D
  Type         : {embeddings.dtype}
  Normalisé L2 : {"Oui ✓" if norme_max - norme_min < 1e-3 else "Non ✗"}
    """)

    if D == 256 and embeddings.dtype == np.float32 and norme_max - norme_min < 1e-3 and N > 0:
        print("  → FICHIER VALIDE — prêt à transmettre à l'équipe classification ✓")
    else:
        print("  → PROBLÈMES DÉTECTÉS — voir les messages ci-dessus ✗")
    ligne("═")


if __name__ == "__main__":
    verifier()