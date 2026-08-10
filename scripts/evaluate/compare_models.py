"""
Script de comparaison — Modèle de base vs Modèle fine-tuné
═══════════════════════════════════════════════════════════════
Compare les deux checkpoints sur les mêmes métriques :
  - Recall@K  (K = 1, 3, 5, 10)
  - Precision@K (K = 1, 5, 10)
  - Accuracy KNN (K = 1, 5)
  - F1-score macro (KNN K=5)
  - Matrice de confusion (KNN K=5)

Les résultats sont affichés en console et sauvegardés dans
  reports/comparison_report.txt (métriques texte)
  reports/confusion_matrix_base.png
  reports/confusion_matrix_finetuned.png

Usage :
    python scripts/evaluate/compare_models.py
    python scripts/evaluate/compare_models.py \\
        --base      models/embedding_model_base.pt \\
        --finetuned models/embedding_model_finetuned.pt \\
        --images    data/processed/images
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Assure que la racine du projet est dans sys.path (pour import src.*)
_ROOT_SYSPATH = Path(__file__).resolve().parents[2]
if str(_ROOT_SYSPATH) not in sys.path:
    sys.path.insert(0, str(_ROOT_SYSPATH))

import matplotlib
matplotlib.use("Agg")          # pas d'affichage interactif
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Chemins par défaut
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
DEFAULT_BASE           = ROOT / "models" / "embedding_model_base.pt"
DEFAULT_FINETUNED      = ROOT / "models" / "embedding_model_finetuned.pt"
DEFAULT_IMAGES         = ROOT / "data" / "processed" / "images"
DEFAULT_EMBEDDINGS_H5  = ROOT / "data" / "processed" / "embeddings" / "embeddings.h5"
REPORT_DIR             = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

NOM_CLASSES = {
    0: "Alloco",
    1: "Foutou",
    2: "Kedjenou",
    3: "Mafé",
    4: "Thiéboudiene",
    5: "Yassa poulet",
}
N_CLASSES = len(NOM_CLASSES)
CLASS_NAMES = [NOM_CLASSES[i] for i in range(N_CLASSES)]


# ---------------------------------------------------------------------------
# Helpers : métriques retrieval
# ---------------------------------------------------------------------------

def recall_at_k(embeddings: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Recall@K global : au moins 1 voisin du même plat parmi les K plus proches."""
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -999.0)
    succes = 0
    for i in range(len(embeddings)):
        top_k = np.argsort(sims[i])[::-1][:k]
        if labels[i] in labels[top_k]:
            succes += 1
    return succes / len(embeddings)


def precision_at_k(embeddings: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Precision@K global : fraction moyenne des K voisins du bon plat."""
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -999.0)
    total = 0.0
    for i in range(len(embeddings)):
        top_k = np.argsort(sims[i])[::-1][:k]
        total += np.sum(labels[top_k] == labels[i]) / k
    return total / len(embeddings)


def recall_per_class(
    embeddings: np.ndarray, labels: np.ndarray, k: int
) -> dict[int, float]:
    """Recall@K par classe."""
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -999.0)
    buckets: dict[int, list[int]] = {}
    for i in range(len(embeddings)):
        top_k = np.argsort(sims[i])[::-1][:k]
        cls = int(labels[i])
        buckets.setdefault(cls, []).append(int(labels[i] in labels[top_k]))
    return {cls: float(np.mean(v)) for cls, v in buckets.items()}


def knn_predict(
    embeddings: np.ndarray, labels: np.ndarray, k: int
) -> np.ndarray:
    """
    Prédiction KNN leave-one-out (vote majoritaire parmi les K voisins).
    Retourne un tableau de prédictions de même taille que labels.
    """
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -999.0)
    preds = []
    for i in range(len(embeddings)):
        top_k = np.argsort(sims[i])[::-1][:k]
        votes = labels[top_k]
        # vote majoritaire
        unique, counts = np.unique(votes, return_counts=True)
        preds.append(int(unique[np.argmax(counts)]))
    return np.array(preds, dtype=np.int32)


# ---------------------------------------------------------------------------
# Chargement depuis HDF5
# ---------------------------------------------------------------------------

def load_embeddings_h5(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Lit un fichier embeddings.h5 et retourne (embeddings, labels)."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        embeddings = f["embeddings"][:]
        if "labels" not in f:
            print(f"  ✗ Pas de labels dans {h5_path}")
            sys.exit(1)
        labels = f["labels"][:].astype(np.int32)
    print(f"  HDF5 chargé : {h5_path}  ({len(embeddings)} vecteurs, dim={embeddings.shape[1]})")
    return embeddings.astype(np.float32), labels


# ---------------------------------------------------------------------------
# Génération des embeddings depuis les images
# ---------------------------------------------------------------------------

def generate_embeddings(
    checkpoint_path: Path,
    images_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge un checkpoint, parcourt le dataset et retourne (embeddings, labels).
    Si le checkpoint n'existe pas, utilise un modèle initialisé depuis la config.
    """
    from src.config.settings import embed_cfg
    from src.process.embedding_model import FoodEmbeddingModel, resolve_device
    from src.utils.image_utils import FoodImageDataset, get_inference_transforms

    device = resolve_device(embed_cfg.device)

    if checkpoint_path.exists():
        model = FoodEmbeddingModel.load(checkpoint_path)
        print(f"  Checkpoint chargé : {checkpoint_path}")
    else:
        model = FoodEmbeddingModel.from_config(freeze_backbone=True)
        print(f"  ⚠ Checkpoint introuvable ({checkpoint_path}), modèle config utilisé.")

    model = model.to(device)
    model.eval()

    transform = get_inference_transforms()
    dataset = FoodImageDataset(root_dir=images_dir, transform=transform)

    if len(dataset) == 0:
        print(f"  ✗ Aucune image dans {images_dir}")
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size=embed_cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    all_emb: list[np.ndarray] = []
    all_lbl: list[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="    Embeddings", unit="batch", leave=False):
            imgs = batch["image"].to(device, non_blocking=True)
            emb = model.encode(imgs, return_numpy=True)
            all_emb.append(emb)
            if "label" in batch:
                all_lbl.extend(batch["label"].tolist())

    embeddings = np.vstack(all_emb).astype(np.float32)
    labels = np.array(all_lbl, dtype=np.int32)

    if len(labels) == 0:
        print("  ✗ Aucun label détecté — structure de dossiers incorrecte ?")
        sys.exit(1)

    return embeddings, labels


# ---------------------------------------------------------------------------
# Visualisation : matrice de confusion
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Sauvegarde une matrice de confusion normalisée en PNG."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="Classe réelle",
        xlabel="Classe prédite",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = int(cm[i, j])
            pct   = cm_norm[i, j]
            color = "white" if pct > thresh else "black"
            ax.text(
                j, i,
                f"{count}\n({pct:.0%})",
                ha="center", va="center",
                fontsize=8, color=color,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Matrice de confusion → {output_path}")


# ---------------------------------------------------------------------------
# Évaluation d'un seul modèle
# ---------------------------------------------------------------------------

def evaluate_model(
    name: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_eval: int = 2000,
) -> dict:
    """
    Calcule toutes les métriques pour un jeu d'embeddings.
    Retourne un dict avec les scores.
    """
    N = len(embeddings)

    # Sous-échantillonnage si nécessaire (garder la repr. de toutes les classes)
    if N > max_eval:
        rng = np.random.default_rng(42)
        idx = []
        classes = np.unique(labels)
        per_class = max_eval // len(classes)
        for c in classes:
            c_idx = np.where(labels == c)[0]
            chosen = rng.choice(c_idx, min(per_class, len(c_idx)), replace=False)
            idx.extend(chosen.tolist())
        # compléter si quota non atteint
        remaining = max_eval - len(idx)
        if remaining > 0:
            all_idx = set(range(N))
            pool = list(all_idx - set(idx))
            extra = rng.choice(pool, min(remaining, len(pool)), replace=False)
            idx.extend(extra.tolist())
        idx = np.array(idx)
        embeddings = embeddings[idx]
        labels     = labels[idx]
        N = len(embeddings)

    print(f"\n  [{name}] — {N} images, dim={embeddings.shape[1]}")

    t0 = time.perf_counter()

    # --- Recall@K & Precision@K ---
    recall = {k: recall_at_k(embeddings, labels, k) for k in [1, 3, 5, 10]}
    precision = {k: precision_at_k(embeddings, labels, k) for k in [1, 5, 10]}
    recall_cls = recall_per_class(embeddings, labels, k=5)

    # --- KNN classification (K=1 et K=5) ---
    preds_k1 = knn_predict(embeddings, labels, k=1)
    preds_k5 = knn_predict(embeddings, labels, k=5)

    acc_k1 = accuracy_score(labels, preds_k1)
    acc_k5 = accuracy_score(labels, preds_k5)
    f1_k5  = f1_score(labels, preds_k5, average="macro", zero_division=0)
    cm_k5  = confusion_matrix(labels, preds_k5, labels=list(range(N_CLASSES)))
    report = classification_report(
        labels, preds_k5,
        labels=list(range(N_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    elapsed = time.perf_counter() - t0

    return {
        "name": name,
        "n_samples": N,
        "dim": embeddings.shape[1],
        "recall": recall,
        "precision": precision,
        "recall_per_class": recall_cls,
        "acc_k1": acc_k1,
        "acc_k5": acc_k5,
        "f1_k5": f1_k5,
        "cm_k5": cm_k5,
        "report": report,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Affichage console
# ---------------------------------------------------------------------------

def ligne(char: str = "─", n: int = 70) -> None:
    print(char * n)


def print_results(res: dict) -> None:
    name = res["name"]
    print(f"\n{'═' * 70}")
    print(f"  RÉSULTATS — {name}")
    print(f"{'═' * 70}")
    print(f"  Images évaluées : {res['n_samples']}   Dim : {res['dim']}D   "
          f"Temps : {res['elapsed']:.1f}s")

    print("\n  Recall@K global")
    ligne()
    seuils = {1: 0.50, 3: 0.65, 5: 0.75, 10: 0.85}
    for k, score in res["recall"].items():
        barre  = "█" * int(score * 20)
        seuil  = seuils.get(k, 0.70)
        statut = "✓ BON" if score >= seuil else ("▲ ACCEPTABLE" if score >= seuil - 0.15 else "✗ FAIBLE")
        print(f"  Recall@{k:2d}  {score:6.1%}  {barre:<20}  {statut}")

    print("\n  Precision@K global")
    ligne()
    for k, score in res["precision"].items():
        print(f"  Precision@{k:2d}  {score:6.1%}")

    print("\n  Recall@5 par classe")
    ligne()
    for cls_idx in sorted(res["recall_per_class"]):
        score = res["recall_per_class"][cls_idx]
        nom   = NOM_CLASSES.get(cls_idx, f"classe_{cls_idx}")
        barre = "█" * int(score * 15)
        print(f"  {nom:20s}  {score:6.1%}  {barre}")

    print("\n  Classification KNN (leave-one-out)")
    ligne()
    print(f"  Accuracy KNN-1  : {res['acc_k1']:.1%}")
    print(f"  Accuracy KNN-5  : {res['acc_k5']:.1%}")
    print(f"  F1-score macro  : {res['f1_k5']:.1%}  (KNN-5)")

    print("\n  Rapport par classe (KNN-5)")
    ligne()
    print(res["report"])


def print_comparison(base: dict, ft: dict) -> None:
    print(f"\n{'═' * 70}")
    print("  TABLEAU COMPARATIF : BASE vs FINE-TUNÉ")
    print(f"{'═' * 70}")

    header = f"  {'Métrique':30s}  {'Base':>10}  {'Fine-tuné':>10}  {'Delta':>8}"
    print(header)
    ligne()

    def row(label: str, v_base: float, v_ft: float) -> None:
        delta = v_ft - v_base
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:30s}  {v_base:>9.1%}  {v_ft:>9.1%}  {sign}{delta:>7.1%}")

    row("Recall@1",         base["recall"][1],  ft["recall"][1])
    row("Recall@3",         base["recall"][3],  ft["recall"][3])
    row("Recall@5",         base["recall"][5],  ft["recall"][5])
    row("Recall@10",        base["recall"][10], ft["recall"][10])
    ligne()
    row("Precision@1",      base["precision"][1],  ft["precision"][1])
    row("Precision@5",      base["precision"][5],  ft["precision"][5])
    row("Precision@10",     base["precision"][10], ft["precision"][10])
    ligne()
    row("Accuracy KNN-1",   base["acc_k1"],  ft["acc_k1"])
    row("Accuracy KNN-5",   base["acc_k5"],  ft["acc_k5"])
    row("F1-score macro",   base["f1_k5"],   ft["f1_k5"])
    ligne()

    # Recommandation automatique
    wins_ft = sum([
        ft["recall"][5]    > base["recall"][5],
        ft["recall"][1]    > base["recall"][1],
        ft["acc_k5"]       > base["acc_k5"],
        ft["f1_k5"]        > base["f1_k5"],
        ft["precision"][5] > base["precision"][5],
    ])
    print(f"\n  Le modèle fine-tuné gagne sur {wins_ft}/5 métriques clés.")
    if wins_ft >= 4:
        print("  → RECOMMANDATION : utiliser le modèle FINE-TUNÉ en phase 3.")
    elif wins_ft >= 2:
        print("  → RECOMMANDATION : le fine-tuning apporte un gain marginal.")
        print("    Évaluer le coût de ré-entraînement vs le gain obtenu.")
    else:
        print("  → RECOMMANDATION : le modèle BASE est suffisant pour la phase 3.")
    print(f"{'═' * 70}")


# ---------------------------------------------------------------------------
# Sauvegarde rapport texte
# ---------------------------------------------------------------------------

def save_report(base: dict, ft: dict, path: Path) -> None:
    """Écrit le rapport comparatif dans un fichier texte."""
    lines: list[str] = []
    sep  = "=" * 70
    dash = "-" * 70

    lines += [
        sep,
        "RAPPORT COMPARATIF — MODÈLE BASE vs FINE-TUNÉ — NUTRI-IA",
        sep,
        f"Date d'exécution   : {time.strftime('%Y-%m-%d %H:%M')}",
        f"Images évaluées    : {base['n_samples']}",
        f"Dimension embeddings : {base['dim']}D",
        f"Classes            : {N_CLASSES} ({', '.join(CLASS_NAMES)})",
        "",
    ]

    for res in [base, ft]:
        lines += [
            dash,
            f"MODÈLE : {res['name']}",
            dash,
            "Recall@K global",
        ]
        for k, score in res["recall"].items():
            lines.append(f"  Recall@{k:2d}  = {score:.1%}")
        lines += ["", "Precision@K global"]
        for k, score in res["precision"].items():
            lines.append(f"  Precision@{k:2d} = {score:.1%}")
        lines += ["", "Recall@5 par classe"]
        for cls_idx in sorted(res["recall_per_class"]):
            nom   = NOM_CLASSES.get(cls_idx, f"classe_{cls_idx}")
            score = res["recall_per_class"][cls_idx]
            lines.append(f"  {nom:20s} : {score:.1%}")
        lines += [
            "",
            "Classification KNN (leave-one-out)",
            f"  Accuracy KNN-1  : {res['acc_k1']:.1%}",
            f"  Accuracy KNN-5  : {res['acc_k5']:.1%}",
            f"  F1-score macro  : {res['f1_k5']:.1%}",
            "",
            "Rapport classification (KNN-5)",
            res["report"],
        ]

    lines += [
        sep,
        "TABLEAU COMPARATIF",
        sep,
        f"  {'Métrique':30s}  {'Base':>9}  {'Fine-tuné':>9}  {'Delta':>8}",
        dash,
    ]
    pairs = [
        ("Recall@1",       base["recall"][1],     ft["recall"][1]),
        ("Recall@3",       base["recall"][3],     ft["recall"][3]),
        ("Recall@5",       base["recall"][5],     ft["recall"][5]),
        ("Recall@10",      base["recall"][10],    ft["recall"][10]),
        ("Precision@1",    base["precision"][1],  ft["precision"][1]),
        ("Precision@5",    base["precision"][5],  ft["precision"][5]),
        ("Precision@10",   base["precision"][10], ft["precision"][10]),
        ("Accuracy KNN-1", base["acc_k1"],        ft["acc_k1"]),
        ("Accuracy KNN-5", base["acc_k5"],        ft["acc_k5"]),
        ("F1-score macro", base["f1_k5"],         ft["f1_k5"]),
    ]
    for label, vb, vf in pairs:
        delta = vf - vb
        sign  = "+" if delta >= 0 else ""
        lines.append(f"  {label:30s}  {vb:>8.1%}  {vf:>8.1%}  {sign}{delta:>7.1%}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  Rapport texte → {path}")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare le modèle de base et le modèle fine-tuné.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base", type=str, default=str(DEFAULT_BASE),
        help="Checkpoint du modèle de base (.pt).",
    )
    parser.add_argument(
        "--finetuned", type=str, default=str(DEFAULT_FINETUNED),
        help="Checkpoint du modèle fine-tuné (.pt).",
    )
    parser.add_argument(
        "--images", type=str, default=str(DEFAULT_IMAGES),
        help="Répertoire des images processed (structure dossiers = classes).",
    )
    parser.add_argument(
        "--embeddings-base", type=str, default=None,
        help="Fichier HDF5 d'embeddings pré-calculés pour le modèle BASE "
             "(utilisé si --images est absent).",
    )
    parser.add_argument(
        "--embeddings-finetuned", type=str, default=None,
        help="Fichier HDF5 d'embeddings pré-calculés pour le modèle FINE-TUNÉ "
             "(utilisé si --images est absent).",
    )
    parser.add_argument(
        "--max-eval", type=int, default=2000,
        help="Nombre max d'images à évaluer (pour limiter le temps de calcul).",
    )
    args = parser.parse_args()

    base_path  = Path(args.base)
    ft_path    = Path(args.finetuned)
    img_dir    = Path(args.images)
    h5_base    = Path(args.embeddings_base) if args.embeddings_base else None
    h5_ft      = Path(args.embeddings_finetuned) if args.embeddings_finetuned else None

    print("═" * 70)
    print("  COMPARAISON MODÈLES — NUTRI-IA")
    print("═" * 70)
    print(f"  Modèle base      : {base_path}")
    print(f"  Modèle fine-tuné : {ft_path}")

    images_available = img_dir.exists() and any(img_dir.iterdir()) if img_dir.exists() else False

    # --- Source des embeddings ---
    # Priorité : (1) images disponibles → re-génération depuis chaque checkpoint
    #            (2) HDF5 explicites fournis en argument
    #            (3) HDF5 par défaut du projet (produit par le dernier `main.py embed`)
    #                → attribué au modèle fine-tuné (dernier lancé en pratique)

    print("\n[1/4] Chargement / génération des embeddings — modèle BASE")
    if images_available:
        print(f"  Images trouvées dans {img_dir} — re-génération depuis le checkpoint.")
        emb_base, lbl_base = generate_embeddings(base_path, img_dir)
    elif h5_base and h5_base.exists():
        emb_base, lbl_base = load_embeddings_h5(h5_base)
    elif DEFAULT_EMBEDDINGS_H5.exists():
        print(f"  Images absentes. Utilisation du HDF5 par défaut pour le modèle BASE.")
        print(f"  ⚠  Attention : ce fichier a peut-être été produit par le modèle fine-tuné.")
        print(f"     Pour une comparaison exacte, fournissez --images ou --embeddings-base.")
        emb_base, lbl_base = load_embeddings_h5(DEFAULT_EMBEDDINGS_H5)
    else:
        print("  ✗ Ni images ni fichier HDF5 disponibles. Lancez d'abord :")
        print("     python main.py embed --checkpoint models/embedding_model_base.pt")
        sys.exit(1)

    print("\n[2/4] Chargement / génération des embeddings — modèle FINE-TUNÉ")
    if images_available:
        emb_ft, lbl_ft = generate_embeddings(ft_path, img_dir)
    elif h5_ft and h5_ft.exists():
        emb_ft, lbl_ft = load_embeddings_h5(h5_ft)
    elif DEFAULT_EMBEDDINGS_H5.exists():
        print(f"  Utilisation du HDF5 par défaut pour le modèle FINE-TUNÉ.")
        emb_ft, lbl_ft = load_embeddings_h5(DEFAULT_EMBEDDINGS_H5)
    else:
        print("  ✗ Ni images ni fichier HDF5 disponibles.")
        sys.exit(1)

    # --- Évaluation ---
    print("\n[3/4] Calcul des métriques...")
    results_base = evaluate_model("Modèle BASE (EfficientNet-B2, backbone gelé)",
                                  emb_base, lbl_base, args.max_eval)
    results_ft   = evaluate_model("Modèle FINE-TUNÉ (Triplet Loss, 20 epochs)",
                                  emb_ft, lbl_ft, args.max_eval)

    # --- Affichage ---
    print_results(results_base)
    print_results(results_ft)
    print_comparison(results_base, results_ft)

    # --- Exports ---
    print("\n[4/4] Sauvegarde des résultats...")
    save_report(results_base, results_ft, REPORT_DIR / "comparison_report.txt")

    plot_confusion_matrix(
        results_base["cm_k5"], CLASS_NAMES,
        title="Matrice de confusion — Modèle BASE (KNN-5)",
        output_path=REPORT_DIR / "confusion_matrix_base.png",
    )
    plot_confusion_matrix(
        results_ft["cm_k5"], CLASS_NAMES,
        title="Matrice de confusion — Modèle FINE-TUNÉ (KNN-5)",
        output_path=REPORT_DIR / "confusion_matrix_finetuned.png",
    )

    print("\n  Fichiers générés :")
    print(f"    reports/comparison_report.txt")
    print(f"    reports/confusion_matrix_base.png")
    print(f"    reports/confusion_matrix_finetuned.png")
    print("═" * 70)


if __name__ == "__main__":
    main()
