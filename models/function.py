import torch
import os
from main import CustomCIFAR10Model



def export_to_onnx(model_path, onnx_path="modelCIFAR10.onnx"):
    """
    Charge un modèle sauvegardé et l'exporte au format ONNX.

    Args:
        model_path (str): Le chemin vers le fichier .pth contenant les poids du modèle.
        onnx_path (str): Le chemin de sortie pour le fichier ONNX.

    Returns:
        None
    """
    # Vérifier que le fichier de poids existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier de modèle {model_path} est introuvable.")

    model = CustomCIFAR10Model()
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("Modèle chargé avec succès.")

    dummy_input = torch.randn(1, 3, 32, 32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=["input"],  # Nom des entrées
        output_names=["output"]  # Nom des sorties
    )
    print(f"Modèle exporté avec succès au format ONNX : {onnx_path}")


if __name__ == "__main__":
    model_path = "modelCIFAR10.pth"
    onnx_path = "modelCIFAR10.onnx"
    export_to_onnx(model_path, onnx_path)