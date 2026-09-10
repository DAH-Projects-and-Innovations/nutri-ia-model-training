"""
Conversion Keras -> PyTorch (.pt) via le pont ONNX (SavedModel -> ONNX -> onnx2torch).

Keras/TensorFlow n'a pas d'export natif vers le format .pt. Les modèles de ces notebooks
contiennent aussi une couche d'augmentation (RandomFlip/RandomRotation/RandomZoom) qui est un
no-op à l'inférence mais qui utilise des opérations TF non supportées par la chaîne
ONNX -> PyTorch (StatelessRandomUniformV2, ImageProjectiveTransformV3, ReverseSequence côté
onnx2torch). On reconstruit donc un modèle d'inférence qui réutilise les MÊMES couches
entraînées (poids identiques, donc sortie identique) mais sans repasser par l'augmentation,
puis on exporte celui-ci.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import onnx
import onnx2torch
import torch
from tensorflow import keras


def _build_inference_model(model, backbone_layer_name, head_layer_names, preprocess_fn, image_size):
    backbone = model.get_layer(backbone_layer_name)
    new_input = keras.Input(shape=(*image_size, 3), name="input")
    x = preprocess_fn(new_input)
    x = backbone(x, training=False)
    for name in head_layer_names:
        layer = model.get_layer(name)
        x = layer(x, training=False) if "dropout" in name.lower() else layer(x)
    return keras.Model(new_input, x)


def export_keras_as_pt(
    model,
    pt_path: str | Path,
    backbone_layer_name: str,
    head_layer_names: Sequence[str],
    preprocess_fn: Callable,
    image_size: tuple[int, int] = (224, 224),
    test_input: np.ndarray | None = None,
    max_allowed_diff: float = 1e-4,
) -> Path:
    """
    Exporte un modèle Keras entraîné en .pt PyTorch (SavedModel -> ONNX -> onnx2torch).

    Si `test_input` est fourni (un batch d'images non pré-traitées, même format que l'entrée
    du modèle Keras), vérifie que la sortie PyTorch correspond à la sortie Keras à
    `max_allowed_diff` près — lève une AssertionError plutôt que de sauvegarder un .pt
    potentiellement incorrect.
    """
    inference_model = _build_inference_model(
        model, backbone_layer_name, head_layer_names, preprocess_fn, image_size
    )

    with tempfile.TemporaryDirectory() as tmp:
        saved_dir = Path(tmp) / "saved_model"
        onnx_path = Path(tmp) / "model.onnx"
        inference_model.export(str(saved_dir))
        subprocess.run(
            ["python", "-m", "tf2onnx.convert",
             "--saved-model", str(saved_dir),
             "--output", str(onnx_path),
             "--opset", "17"],
            check=True, capture_output=True,
        )
        onnx_model = onnx.load(str(onnx_path))
        torch_model = onnx2torch.convert(onnx_model)
    torch_model.eval()

    if test_input is not None:
        keras_out = inference_model.predict(test_input, verbose=0)
        with torch.no_grad():
            torch_out = torch_model(torch.from_numpy(test_input)).numpy()
        diff = float(np.abs(keras_out - torch_out).max())
        assert diff < max_allowed_diff, (
            f"Sortie PyTorch différente de Keras (diff max={diff:.2e}) — export refusé."
        )
        print(f"Vérification Keras vs PyTorch OK — diff max = {diff:.2e}")

    pt_path = Path(pt_path)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_model, pt_path)
    return pt_path
