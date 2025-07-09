import torch
from .hmm import HMM
import pandas as pd
from typing import Tuple

torch.set_float32_matmul_precision('high')


def load_hmm_model(hmm_model_path: str, device: str = 'cuda:0') -> HMM:
    """
    Load the pretrained HMM model.

    Args:
        hmm_model_path (str): Path to the saved HMM model.
        device (str): Device to load the model on.

    Returns:
        HMM: Loaded HMM model.
    """
    hmm_model = HMM.from_pretrained(hmm_model_path, local_files_only=True).to(device)
    hmm_model.eval()
    return hmm_model

def load_weights(weights_file: str, device: str = "cpu") -> torch.Tensor:
    """
    Load weights from CSV file.

    Args:
        weights_file (str): Path to the weights CSV file with columns 'Token ID' and 'Coefficient'.
        device (str): Device to load the tensors onto.

    Returns:
        torch.Tensor: weights_tensor of shape (V,) where V is the vocab size.
    """
    try:
        df_weights = pd.read_csv(weights_file)
    except Exception as e:
        raise FileNotFoundError(f"Failed to read weights file '{weights_file}': {e}")

    required_columns = {'Token ID', 'Coefficient'}
    if not required_columns.issubset(df_weights.columns):
        raise ValueError(f"Weights CSV must contain columns: {required_columns}")

    df_weights = df_weights.sort_values('Token ID').reset_index(drop=True)

    expected_token_ids = list(range(len(df_weights)))
    actual_token_ids = df_weights['Token ID'].tolist()
    if actual_token_ids != expected_token_ids:
        raise ValueError("Token IDs in weights file are not sequential starting from 0.")

    coefficients = df_weights['Coefficient'].values
    weights_tensor = torch.tensor(coefficients, dtype=torch.float32, device=device)

    return weights_tensor
