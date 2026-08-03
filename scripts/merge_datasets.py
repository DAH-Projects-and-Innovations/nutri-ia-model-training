from pathlib import Path
import shutil

dossier_images = Path("data/raw/images")
dossier_aug = Path("data/raw/augmented_images")

for plat_aug in dossier_aug.iterdir():

    if not plat_aug.is_dir():
        continue

    dest = dossier_images / plat_aug.name
    dest.mkdir(exist_ok=True)

    for img in plat_aug.glob("*"):

        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        destination = dest / img.name

        if destination.exists():
            destination = dest / f"aug_{img.name}"

        shutil.copy2(img, destination)

print("Fusion terminée.")