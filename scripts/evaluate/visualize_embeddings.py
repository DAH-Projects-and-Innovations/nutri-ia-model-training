"""
Script 2 — Visualisation t-SNE des embeddings
═══════════════════════════════════════════════════
Réduit les vecteurs 256D en 2D et affiche un graphique.
Si les plats forment des groupes séparés → modèle bon.

Usage :
    python scripts/evaluate/visualize_embeddings.py

Installe d'abord :
    uv add scikit-learn matplotlib
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.manifold import TSNE

# Configuration
EMBEDDINGS_H5 = Path("data/processed/embeddings/embeddings.h5")
REPORT_DIR    = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# Noms des classes 
NOM_CLASSES = {
    0: "Alloco",
    1: "Foutou",
    2: "Kedjenou",
    3: "Mafé",
    4: "Thiéboudiene",
    5: "Yassa poulet",
}

COULEURS = [
    "#E63946",  # rouge
    "#2196F3",  # bleu
    "#4CAF50",  # vert
    "#FF9800",  # orange
    "#9C27B0",  # violet
    "#00BCD4",  # cyan
]


def visualiser():
    print("═" * 60)
    print("  VISUALISATION t-SNE DES EMBEDDINGS")
    print("═" * 60)

    # Chargement des embeddings et des labels 
    print("\n[1] Chargement des embeddings...")
    with h5py.File(EMBEDDINGS_H5, "r") as f:
        embeddings = f["embeddings"][:]
        labels     = f["labels"][:] if "labels" in f else None

    N, D = embeddings.shape
    print(f"  {N} embeddings de dimension {D} chargés")

    if labels is None:
        print("  ▲  Pas de labels — le graphique sera sans couleur par classe")

    # t-SNE : réduction 256D → 2D 
    print("\n[2] Réduction t-SNE (256D → 2D)...")
    print("  Cela peut prendre 1 à 3 minutes selon le nombre d'images...")

    # Sous-échantillonner si trop d'images (t-SNE lent sur > 5000 points)
    MAX_POINTS = 2000
    if N > MAX_POINTS:
        print(f"  Sous-échantillonnage à {MAX_POINTS} images pour la rapidité...")
        idx = np.random.choice(N, MAX_POINTS, replace=False)
        embeddings_vis = embeddings[idx]
        labels_vis     = labels[idx] if labels is not None else None
    else:
        embeddings_vis = embeddings
        labels_vis     = labels

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, len(embeddings_vis) - 1),
        max_iter=1000,
        verbose=1,
    )
    emb_2d = tsne.fit_transform(embeddings_vis)
    print(f"  t-SNE terminé — shape: {emb_2d.shape}")

    # Graphique 
    print("\n[3] Génération du graphique...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Visualisation t-SNE des Embeddings — Nutri-IA\n"
        "Si chaque couleur forme un groupe séparé → modèle d'embedding bon",
        fontsize=13, y=1.01
    )

    # Graphique gauche : par classe 
    ax1 = axes[0]
    ax1.set_title("Par classe (plat)", fontsize=12, pad=10)

    if labels_vis is not None:
        classes = np.unique(labels_vis)
        legendes = []
        for idx_classe in classes:
            masque  = labels_vis == idx_classe
            couleur = COULEURS[int(idx_classe) % len(COULEURS)]
            nom     = NOM_CLASSES.get(int(idx_classe), f"Classe {idx_classe}")
            ax1.scatter(
                emb_2d[masque, 0],
                emb_2d[masque, 1],
                c=couleur,
                alpha=0.65,
                s=18,
                edgecolors="none",
            )
            legendes.append(mpatches.Patch(color=couleur, label=nom))

        ax1.legend(
            handles=legendes,
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
        )
    else:
        ax1.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.5, s=15)

    ax1.set_xlabel("t-SNE dimension 1")
    ax1.set_ylabel("t-SNE dimension 2")
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#F8F9FA")

    # Graphique droit : densité 
    ax2 = axes[1]
    ax2.set_title("Densité globale des embeddings", fontsize=12, pad=10)
    ax2.hexbin(
        emb_2d[:, 0], emb_2d[:, 1],
        gridsize=40,
        cmap="YlOrRd",
        mincnt=1,
    )
    ax2.set_xlabel("t-SNE dimension 1")
    ax2.set_ylabel("t-SNE dimension 2")
    ax2.set_facecolor("#F8F9FA")

    plt.tight_layout()

    # Sauvegarde 
    sortie = REPORT_DIR / "tsne_embeddings.png"
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    print(f"\n  Graphique sauvegardé → {sortie}")
    plt.show()

    # Guide de lecture 
    print("\n" + "═" * 60)
    print("  COMMENT LIRE CE GRAPHIQUE")
    print("═" * 60)
    print("""
  ✓   BON résultat :
     Chaque couleur forme un nuage séparé des autres.
     Ex: tous les ronds rouges (thiéboudienne) sont ensemble,
         tous les bleus (mafé) sont ensemble, etc.
     → Le modèle distingue bien les plats.

  ▲  RÉSULTAT MOYEN :
     Les nuages sont un peu séparés mais avec des mélanges
     aux bords.
     → Acceptable pour un MVP avec peu d'images.

  ✗ MAUVAIS résultat :
     Toutes les couleurs mélangées — un seul gros nuage.
     → Le modèle ne fait pas la différence entre les plats.
     → Causes possibles :
        - Trop peu d'images par plat (< 20)
        - Images de mauvaise qualité
        - Dossiers mal structurés
    """)
    print("═" * 60)


if __name__ == "__main__":
    visualiser()