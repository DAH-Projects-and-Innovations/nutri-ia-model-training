#  scripts/save_base_model.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.process.embedding_model import FoodEmbeddingModel

# Créer le modèle de base (sans fine-tuning)
model = FoodEmbeddingModel.from_config(freeze_backbone=True)

# Sauvegarder
Path("models").mkdir(exist_ok=True)
model.save("models/embedding_model_base.pt")
print("Modèle sauvegardé → models/embedding_model_base.pt")