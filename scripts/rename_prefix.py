from pathlib import Path

base = Path("data/raw/images")

# Correspondance dossier -> préfixe à remplacer
mapping = {
    "alloco": ("IMG_CIV03", "alloco"),
    "mafe": ("IMG_SEN02", "mafe"),
 # adapte si besoin
}

for dossier in base.iterdir():

    if not dossier.is_dir():
        continue

    nom_dossier = dossier.name

    if nom_dossier not in mapping:
        print(f"Aucune règle pour {nom_dossier}")
        continue

    ancien_prefixe, nouveau_prefixe = mapping[nom_dossier]

    print(f"\nTraitement de {nom_dossier}")

    images = list(dossier.glob("*.jpg")) + list(dossier.glob("*.jpeg"))

    for img in images:

        if img.name.startswith(ancien_prefixe):

            nouveau_nom = img.name.replace(
                ancien_prefixe,
                nouveau_prefixe,
                1
            )

            nouveau_fichier = dossier / nouveau_nom

            img.rename(nouveau_fichier)

            print(f"{img.name} -> {nouveau_nom}")

print("\nRenommage terminé.")