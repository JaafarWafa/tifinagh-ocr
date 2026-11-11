#!/usr/bin/env python3
"""
Tifinagh OCR Training Script using DocTR's CRNN model
"""

import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from doctr.io import DocumentFile
from doctr.models import crnn_vgg16_bn
from doctr.datasets import RecognitionDataset
from doctr.transforms import Resize
import torchvision.transforms as T
from tqdm import tqdm
from vocab_config import get_vocab_size, get_char_to_idx, get_idx_to_char
import wandb
import numpy as np
from typing import Dict, List, Tuple

def calculate_accuracy(predictions: List[str], targets: List[str]) -> Tuple[float, float]:
    """
    Calculate exact match and partial match accuracy
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        
    Returns:
        Tuple of (exact_match_accuracy, partial_match_accuracy)
    """
    exact_matches = sum(pred == target for pred, target in zip(predictions, targets))
    exact_match_accuracy = exact_matches / len(targets) if targets else 0.0
    
    # Calculate partial match (character-level accuracy)
    total_chars = 0
    correct_chars = 0
    for pred, target in zip(predictions, targets):
        total_chars += len(target)
        correct_chars += sum(1 for p, t in zip(pred, target) if p == t)
    
    partial_match_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
    
    return exact_match_accuracy, partial_match_accuracy

def load_dataset(data_dir: str, split: str, batch_size: int = 32):
    """Load dataset from the generated data directory"""
    split_dir = Path(data_dir) / split
    annotations_file = split_dir / f"{split}_annotations.json"
    
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    
    # Create dataset with correct transforms
    dataset = RecognitionDataset(
        img_folder=str(split_dir),
        labels_path=str(annotations_file),
        img_transforms=T.Compose([
            Resize((32, 128)),  # Standard size for CRNN
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    # Set the vocabulary for the dataset
    dataset.vocab = list(get_char_to_idx().keys())
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True
    )

def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, targets in pbar:
        images = images.to(device)
        
        # Forward pass
        outputs = model(images, targets)
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images, targets, return_preds=True)
            loss = outputs['loss']
            
            # Get predictions
            if len(outputs["preds"]):
                words, _ = zip(*outputs["preds"])
            else:
                words = []
            all_predictions.extend(words)
            all_targets.extend(targets)
            
            total_loss += loss.item()
    
    # Calculate accuracies
    exact_match_acc, partial_match_acc = calculate_accuracy(all_predictions, all_targets)
    
    return total_loss / len(val_loader), exact_match_acc, partial_match_acc

def main():
    parser = argparse.ArgumentParser(description="Train CRNN model for Tifinagh OCR")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing generated data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='tifinagh-ocr', help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
    
    args = parser.parse_args()
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    train_loader = load_dataset(args.data_dir, "train", args.batch_size)
    val_loader = load_dataset(args.data_dir, "val", args.batch_size)
    
    # Get vocabulary size from vocab_config
    vocab_size = get_vocab_size()
    print(f"Using vocabulary size: {vocab_size}")
    
    # Initialize model with correct parameters
    model = crnn_vgg16_bn(
        pretrained=True,
        vocab=list(get_char_to_idx().keys())
    )
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Training loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss, val_exact_acc, val_partial_acc = evaluate(model, val_loader, device)
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation exact match accuracy: {val_exact_acc:.4f}")
        print(f"Validation partial match accuracy: {val_partial_acc:.4f}")
        
        # Log metrics to wandb
        if args.use_wandb:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_exact_acc': val_exact_acc,
                'val_partial_acc': val_partial_acc,
                'epoch': epoch
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_exact_acc': val_exact_acc,
                'val_partial_acc': val_partial_acc,
                'vocab_size': vocab_size,
                'char_to_idx': get_char_to_idx(),
                'idx_to_char': get_idx_to_char()
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # Save latest model
        latest_path = output_dir / "latest_model.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_exact_acc': val_exact_acc,
            'val_partial_acc': val_partial_acc,
            'vocab_size': vocab_size,
            'char_to_idx': get_char_to_idx(),
            'idx_to_char': get_idx_to_char()
        }, latest_path)
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 