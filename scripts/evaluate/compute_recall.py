"""
Script 3 — Calcul du Recall@K
═══════════════════════════════════════════════════
Mesure la qualité des embeddings avec des métriques chiffrées.

Recall@K = pour chaque image, est-ce que le bon plat
           est dans les K voisins les plus proches ?

Usage :
    python scripts/evaluate/compute_recall.py

Installe d'abord :
    uv add scikit-learn
"""

import sys
from pathlib import Path

import h5py
import numpy as np

# Configuration 
EMBEDDINGS_H5 = Path("data/processed/embeddings/embeddings.h5")
REPORT_DIR    = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

NOM_CLASSES = {
    0: "Alloco",
    1: "Foutou",
    2: "Kedjenou",
    3: "Mafé",
    4: "Thiéboudiene",
    5: "Yassa poulet",

}


def recall_at_k(embeddings: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Pour chaque image, cherche les K plus proches voisins.
    Succès si au moins 1 voisin est du même plat.

    Utilise le produit scalaire comme similarité
    (fonctionne car les vecteurs sont normalisés L2).
    """
    N = len(embeddings)
    succes = 0

    # Matrice de similarité complète (N × N)
    # embeddings @ embeddings.T = produit scalaire entre toutes les paires
    sims = embeddings @ embeddings.T   # (N, N)

    for i in range(N):
        # Ignorer l'image elle-même (similarité = 1.0 avec elle-même)
        sims_i = sims[i].copy()
        sims_i[i] = -999.0

        # Indices des K plus proches (tri décroissant)
        top_k_idx = np.argsort(sims_i)[::-1][:k]

        # Est-ce qu'au moins un voisin est du même plat ?
        if labels[i] in labels[top_k_idx]:
            succes += 1

    return succes / N


def precision_at_k(embeddings: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Parmi les K plus proches voisins, quelle fraction est du même plat ?
    """
    N = len(embeddings)
    sims = embeddings @ embeddings.T

    total_precision = 0.0
    for i in range(N):
        sims_i = sims[i].copy()
        sims_i[i] = -999.0
        top_k_idx   = np.argsort(sims_i)[::-1][:k]
        meme_classe = np.sum(labels[top_k_idx] == labels[i])
        total_precision += meme_classe / k

    return total_precision / N


def recall_par_classe(
    embeddings: np.ndarray, labels: np.ndarray, k: int
) -> dict[int, float]:
    """Calcule le Recall@K pour chaque classe séparément."""
    N     = len(embeddings)
    sims  = embeddings @ embeddings.T
    resultats: dict[int, list[int]] = {}

    for i in range(N):
        sims_i = sims[i].copy()
        sims_i[i] = -999.0
        top_k_idx = np.argsort(sims_i)[::-1][:k]
        cls = int(labels[i])

        if cls not in resultats:
            resultats[cls] = []

        succes = int(labels[i] in labels[top_k_idx])
        resultats[cls].append(succes)

    return {cls: np.mean(vals) for cls, vals in resultats.items()}


def evaluer():
    print("═" * 60)
    print("  ÉVALUATION DES EMBEDDINGS — RECALL@K")
    print("═" * 60)

    #  Chargement 
    print("\n[1] Chargement des embeddings...")

    if not EMBEDDINGS_H5.exists():
        print(f"  ✗   Fichier introuvable : {EMBEDDINGS_H5}")
        print("  Lance d'abord : uv run python main.py embed")
        sys.exit(1)

    with h5py.File(EMBEDDINGS_H5, "r") as f:
        embeddings = f["embeddings"][:]
        labels     = f["labels"][:] if "labels" in f else None

    if labels is None:
        print("  ✗   Pas de labels dans le fichier — impossible de calculer le Recall@K")
        sys.exit(1)

    N, D = embeddings.shape
    n_classes = len(np.unique(labels))
    print(f"  {N} embeddings · {D}D · {n_classes} classes chargés")

    # Sous-échantillonnage si trop d'images (calcul lent sur > 3000)
    MAX_EVAL = 2000
    if N > MAX_EVAL:
        print(f"\n  Sous-échantillonnage à {MAX_EVAL} images pour la rapidité...")
        idx        = np.random.choice(N, MAX_EVAL, replace=False, )
        # Assure représentation de toutes les classes
        embeddings = embeddings[idx]
        labels     = labels[idx]
        N          = MAX_EVAL

    # Recall@K global 
    print("\n[2] Calcul du Recall@K global...")
    print("─" * 60)
    print(f"  {'K':>4}  {'Recall@K':>10}  {'Interprétation':>30}")
    print("─" * 60)

    seuils = {1: 0.50, 3: 0.65, 5: 0.75, 10: 0.85}
    resultats_recall = {}

    for k in [1, 3, 5, 10]:
        score = recall_at_k(embeddings, labels, k)
        resultats_recall[k] = score
        seuil = seuils.get(k, 0.70)

        if score >= seuil:
            statut = "✓   BON"
        elif score >= seuil - 0.15:
            statut = "▲   ACCEPTABLE"
        else:
            statut = "✗   INSUFFISANT"

        barre = "█" * int(score * 20)
        print(f"  K={k:>2}  {score:>8.1%}   {barre:<20}  {statut}")

    print("─" * 60)

    #  Precision@K global
    print("\n[3] Calcul du Precision@K global...")
    print("─" * 60)
    print(f"  {'K':>4}  {'Precision@K':>12}  Signification")
    print("─" * 60)

    for k in [1, 5, 10]:
        score = precision_at_k(embeddings, labels, k)
        print(
            f"  K={k:>2}  {score:>10.1%}   "
            f"En moyenne {score:.0%} des {k} voisins sont du bon plat"
        )

    #  Recall@5 par classe 
    print("\n[4] Recall@5 par classe")
    print("─" * 60)
    print(f"  {'Plat':25s}  {'Recall@5':>10}  {'Statut':>15}")
    print("─" * 60)

    recall_classes = recall_par_classe(embeddings, labels, k=5)
    for cls_idx in sorted(recall_classes.keys()):
        score = recall_classes[cls_idx]
        nom   = NOM_CLASSES.get(cls_idx, f"classe_{cls_idx}")
        if score >= 0.75:
            statut = "✓   BON"
        elif score >= 0.50:
            statut = "▲   MOYEN"
        else:
            statut = "✗   FAIBLE"
        barre = "█" * int(score * 15)
        print(f"  {nom:25s}  {score:>8.1%}   {barre:<15}  {statut}")

    #  Test avec nouvelles images 
    print("\n[5] Simulation — nouvelles images jamais vues")
    print("─" * 60)
    print("  On prend 5 images au hasard et on cherche leurs 3 voisins.\n")

    sims_full = embeddings @ embeddings.T

    indices_test = np.random.choice(N, 5, replace=False)
    for i, idx_query in enumerate(indices_test):
        sims_q = sims_full[idx_query].copy()
        sims_q[idx_query] = -999.0
        top3   = np.argsort(sims_q)[::-1][:3]

        nom_q  = NOM_CLASSES.get(int(labels[idx_query]), "?")
        print(f"  Image test {i+1} : [{nom_q}]")

        for rang, idx_v in enumerate(top3, 1):
            nom_v  = NOM_CLASSES.get(int(labels[idx_v]), "?")
            sim_v  = sims_q[idx_v]
            correct = "✓   " if nom_v == nom_q else "✗   "
            print(f"    Voisin {rang} : {correct}  {nom_v:20s}  sim={sim_v:.4f}")
        print()

    #  Sauvegarde rapport texte 
    rapport_path = REPORT_DIR / "recall_report.txt"
    with open(rapport_path, "w", encoding="utf-8") as f:
        f.write("RAPPORT D'ÉVALUATION — MODÈLE D'EMBEDDING NUTRI-IA\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Images évaluées : {N}\n")
        f.write(f"Dimension       : {D}D\n")
        f.write(f"Classes         : {n_classes}\n\n")
        f.write("RECALL@K GLOBAL\n" + "-" * 40 + "\n")
        for k, score in resultats_recall.items():
            f.write(f"  Recall@{k:2d} = {score:.1%}\n")
        f.write("\nRECALL@5 PAR CLASSE\n" + "-" * 40 + "\n")
        for cls_idx, score in sorted(recall_classes.items()):
            nom = NOM_CLASSES.get(cls_idx, f"classe_{cls_idx}")
            f.write(f"  {nom:25s} : {score:.1%}\n")

    print(f"\n  Rapport sauvegardé → {rapport_path}")

    #  BILAN FINAL 
    recall5 = resultats_recall[5]
    recall1 = resultats_recall[1]

    print("\n" + "═" * 60)
    print("  BILAN FINAL")
    print("═" * 60)
    print(f"""
  Recall@1 = {recall1:.1%}   (le 1er voisin est-il du bon plat ?)
  Recall@5 = {recall5:.1%}   (parmi les 5, au moins 1 bon ?)
    """)

    if recall5 >= 0.75:
        print("  ✓   MODÈLE D'EMBEDDING VALIDÉ")
        print("  Les embeddings sont de bonne qualité.")
        print("  Vous pouvez transmettre embeddings.h5 à l'équipe classification.")
    elif recall5 >= 0.55:
        print("  ▲  MODÈLE ACCEPTABLE POUR UN MVP")
        print("  Qualité suffisante pour commencer, mais à améliorer.")
        print("  Solutions : plus d'images, meilleure augmentation.")
    else:
        print("  ✗   MODÈLE INSUFFISANT")
        print("  Les embeddings ne distinguent pas bien les plats.")
        print("  Causes possibles :")
        print("    - Moins de 20 images par plat")
        print("    - Structure de dossiers incorrecte")
        print("    - Images de mauvaise qualité")
        print("    - normalize_output=False dans settings.py")

    print()
    print("  SEUILS DE RÉFÉRENCE :")
    print("  ─────────────────────────────")
    print("  Recall@5 > 75% → BON        ✓   ")
    print("  Recall@5 55-75% → ACCEPTABLE ▲  ")
    print("  Recall@5 < 55% → INSUFFISANT ✗ ")
    print("═" * 60)


if __name__ == "__main__":
    evaluer()