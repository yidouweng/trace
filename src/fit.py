#!/usr/bin/env python
"""
Fit a linear model to predict attribute scores from token counts.
Always enforces negative coefficients and zero bias using Lasso regression.
Works with toxicity or any custom attribute scored with score_attribute.py.
"""

import os
import sys
import json
import time
import argparse
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.special import logit, expit 
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from transformers import GPT2Tokenizer

# Determine project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

def load_attribute_data(jsonl_path: str, attribute: str = "toxicity") -> pd.DataFrame:
    """Load and parse attribute data from JSONL file."""
    print(f"Loading data from {jsonl_path}...")
    print(f"Target attribute: {attribute}")
    
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON on line {line_num}")
                continue

    df = pd.json_normalize(data)
    print(f"Loaded {len(df)} samples")
    
    # Check if the specified attribute exists in the data
    prompt_attr_col = f"prompt.{attribute}"
    cont_attr_col = f"continuation.{attribute}"
    
    if prompt_attr_col not in df.columns or cont_attr_col not in df.columns:
        print(f"❌ Attribute '{attribute}' not found in data!")
        print(f"Available attributes: {[col.split('.')[1] for col in df.columns if '.' in col and col.startswith(('prompt.', 'continuation.'))]}")
        sys.exit(1)
    
    return df

def preprocess_scores(toxicity_scores: np.ndarray, b: float, c: float) -> tuple:
    """
    Preprocess toxicity scores with logit transformation.
    
    Args:
        toxicity_scores: Raw toxicity scores
        b: Scaling factor for logit transformation
        c: Shift factor for logit transformation
    
    Returns:
        (transformed_scores, log_transformed_scores, original_scores)
    """
    print("Preprocessing toxicity scores...")

    # Invert toxicity scores (1 - toxicity)
    inverted_scores = 1 - toxicity_scores
    original_scores = inverted_scores.copy()

    # Clip to avoid log(0) and log(1)
    epsilon = 1e-15
    clipped_scores = np.clip(inverted_scores, epsilon, 1 - epsilon)
    
    # Apply logit transformation: b * (logit(score) - c)
    logit_scores = logit(clipped_scores)
    modified_logit_scores = b * (logit_scores - c)

    # Transform back to probability space
    transformed_scores = expit(modified_logit_scores)
    
    # Prepare target for regression (log-transformed)
    clipped_transformed = np.clip(transformed_scores, epsilon, 1 - epsilon)
    log_transformed_scores = np.log(clipped_transformed)
    
    print(f"Original mean: {np.mean(original_scores):.4f}")
    print(f"Transformed mean: {np.mean(transformed_scores):.4f}")
    
    return transformed_scores, log_transformed_scores, original_scores

def create_token_matrix(texts: list, tokenizer) -> tuple:
    """
    Create sparse count matrix from tokenized texts.
    
    Returns:
        (count_matrix, token_id_to_index_mapping)
    """
    print("Creating token count matrix...")
    
    # Tokenize all texts
    tokenized_texts = [tokenizer.encode(text, add_special_tokens=False) for text in texts]

    # Get unique token IDs
    unique_token_ids = sorted(set(token_id for tokens in tokenized_texts for token_id in tokens))
    token_id_to_index = {token_id: idx for idx, token_id in enumerate(unique_token_ids)}

    print(f"Found {len(unique_token_ids)} unique tokens")
    
    # Create count matrix
    X = lil_matrix((len(tokenized_texts), len(unique_token_ids)), dtype=int)
    for i, tokens in enumerate(tokenized_texts):
        for token_id in tokens:
            if token_id in token_id_to_index:
                X[i, token_id_to_index[token_id]] += 1
    
    X = X.tocsr()  # Convert to efficient format
    print(f"Count matrix shape: {X.shape}")
    
    return X, token_id_to_index

def fit_lasso_model(X, y: np.ndarray, alpha: float) -> tuple:
    """
    Fit Lasso model with negative coefficient constraint and zero bias.
    
    Returns:
        (coefficients, fitted_log_scores)
    """
    print(f"Fitting Lasso model with alpha={alpha}...")
    start_time = time.time()
    
    # Fit Lasso with positive constraint (we'll negate later)
    # No intercept (enforce zero bias)
    regressor = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=10000)
    regressor.fit(X, -y)  # Fit to -y to enforce negative coefficients
    
    # Get coefficients and negate to enforce negative constraint
    coefficients = -regressor.coef_
    
    # Compute fitted scores
    fitted_log_scores = X.dot(coefficients)
    
    end_time = time.time()
    print(f"Model fitted in {end_time - start_time:.2f} seconds")
    print(f"Number of non-zero coefficients: {np.sum(coefficients != 0)}")
    print(f"Sample coefficients: {coefficients[:10]}")
    
    return coefficients, fitted_log_scores

def save_coefficients(coefficients: np.ndarray, token_id_to_index: dict, vocab_size: int, output_path: str):
    """Save full vocabulary coefficients to CSV file."""
    print(f"Saving coefficients to {output_path}...")
    
    # Initialize full vocabulary with zeros
    full_coefficients = np.zeros(vocab_size)
    
    # Assign learned coefficients to corresponding token IDs
    for token_id, idx in token_id_to_index.items():
        full_coefficients[token_id] = coefficients[idx]

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Token ID", "Coefficient"])
        for token_id in range(vocab_size):
            writer.writerow([token_id, full_coefficients[token_id]])
    
    print(f"✓ Coefficients saved to {output_path}")

def evaluate_model(true_scores: np.ndarray, fitted_scores: np.ndarray, 
                  log_true_scores: np.ndarray, fitted_log_scores: np.ndarray):
    """Evaluate model performance with various metrics."""
    print("\nModel Performance:")
    print("-" * 40)
    
    mse = mean_squared_error(true_scores, fitted_scores)
    mse_log = mean_squared_error(log_true_scores, fitted_log_scores)
    r2 = r2_score(log_true_scores, fitted_log_scores)
    
    print(f"MSE (original space): {mse:.6f}")
    print(f"MSE (log space): {mse_log:.6f}")
    print(f"R² (log space): {r2:.6f}")

def create_diagnostic_plot(original_scores: np.ndarray, transformed_scores: np.ndarray, 
                          fitted_scores: np.ndarray, b: float, c: float, alpha: float, attribute: str = "toxicity"):
    """Create a diagnostic plot showing original, transformed, and fitted distributions."""
    plt.figure(figsize=(12, 6))
    plt.rcParams.update({'font.size': 12})
    
    bins = np.linspace(0, 1, 101)
    
    plt.hist(original_scores, bins=bins, alpha=0.6, label='Original', density=True, 
             color='red', edgecolor='black', linewidth=0.5)
    plt.hist(transformed_scores, bins=bins, alpha=0.5, label='Transformed', density=True, 
             color='blue', edgecolor='black', linewidth=0.5)
    plt.hist(fitted_scores, bins=bins, alpha=0.4, label='Fitted', density=True, 
             color='green', edgecolor='black', linewidth=0.5)
    
    plt.xlabel(f'{attribute.title()} Score')
    plt.ylabel('Density')
    plt.title(f'{attribute.title()} Score Distributions (b={b}, c={c}, α={alpha})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(PROJECT_ROOT, f"results/fit_diagnostic_b{b}_c{c}_alpha{alpha}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Diagnostic plot saved to {plot_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Fit attribute prediction model with Lasso regression.")
    parser.add_argument("--data_path", type=str, 
                        default="data/RTP_train.jsonl",
                        help="Path to training data JSONL file")
    parser.add_argument("--attribute", type=str, default="toxicity",
                        help="Attribute to predict (toxicity, politics, etc.)")
    parser.add_argument("--b", type=float, default=10.0, 
                        help="Scaling factor for logit transformation")
    parser.add_argument("--c", type=float, default=3.0, 
                        help="Shift factor for logit transformation")
    parser.add_argument("--alpha", type=float, default=1e-6, 
                        help="Lasso regularization strength")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path for coefficients (default: data/coefficients_{attribute}.csv)")
    
    args = parser.parse_args()

    # Set default output path based on attribute
    if args.output_path is None:
        args.output_path = f"data/coefficients_{args.attribute}.csv"

    # Make paths absolute
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(PROJECT_ROOT, args.data_path)
    if not os.path.isabs(args.output_path):
        args.output_path = os.path.join(PROJECT_ROOT, args.output_path)
    
    print(f"{args.attribute.title()} Model Fitting")
    print("=" * 50)
    print(f"Data: {args.data_path}")
    print(f"Attribute: {args.attribute}")
    print(f"Transformation: b={args.b}, c={args.c}")
    print(f"Regularization: α={args.alpha}")
    print(f"Output: {args.output_path}")
    print()

    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # Load and preprocess data
    df = load_attribute_data(args.data_path, args.attribute)
    attribute_scores = df[f"continuation.{args.attribute}"].values.astype(float)
    texts = df["continuation.text"].tolist()
    
    transformed_scores, log_scores, original_scores = preprocess_scores(
        attribute_scores, args.b, args.c
    )
    
    # Create token matrix
    X, token_id_to_index = create_token_matrix(texts, tokenizer)
    
    # Fit model
    coefficients, fitted_log_scores = fit_lasso_model(X, log_scores, args.alpha)
    fitted_scores = np.exp(fitted_log_scores)
    
    # Save results
    save_coefficients(coefficients, token_id_to_index, vocab_size, args.output_path)
    
    # Evaluate model
    evaluate_model(transformed_scores, fitted_scores, log_scores, fitted_log_scores)
    
    # Create diagnostic plot
    create_diagnostic_plot(original_scores, transformed_scores, fitted_scores, 
                          args.b, args.c, args.alpha, args.attribute)

    print("\n✓ Model fitting completed successfully!")

if __name__ == "__main__":
    main()