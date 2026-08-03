from pathlib import Path

base = Path("data/raw/images")

for dossier in sorted(base.iterdir()):
    if dossier.is_dir():
        nb = len(list(dossier.glob("*.*")))
        print(f"{dossier.name}: {nb} images")