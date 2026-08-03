from pathlib import Path

ROOT = Path("data/raw/images")

supprimes = 0

for dossier in ROOT.iterdir():
    if not dossier.is_dir():
        continue

    for img in dossier.iterdir():
        if not img.is_file():
            continue

        nom = img.name

        # Cas 1 : images mafe provenant de IMG_SEN...
        if dossier.name.lower() == "mafe" and nom.startswith("IMG_"):
            img.unlink()
            print(f"Supprimé : {img}")
            supprimes += 1
            continue

        # Cas 2 : images alloco provenant de IMG_CIV...
        if dossier.name.lower() == "alloco" and nom.startswith("IMG_"):
                    img.unlink()
                    print(f"Supprimé : {img}")
                    supprimes += 1
                    continue

        # Cas 2 : images renommées en aug_kedjenou...
        if dossier.name.lower() == "kedjenou" and nom.startswith("aug_kedjenou"):
            img.unlink()
            print(f"Supprimé : {img}")
            supprimes += 1
            continue

        # Cas 3 : images renommées en aug_foutou...
        if dossier.name.lower() == "foutou" and nom.startswith("aug_foutou"):
            img.unlink()
            print(f"Supprimé : {img}")
            supprimes += 1

print(f"\n{supprimes} fichiers supprimés.")