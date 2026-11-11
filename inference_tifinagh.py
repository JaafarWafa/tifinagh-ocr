#!/usr/bin/env python3
"""
Tifinagh OCR Inference Script
Loads trained CRNN + VGG16-BN model and performs text recognition on images
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Union, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

# Import model and vocabulary
from train_tifinagh_crnn import CRNN, decode_predictions
from vocab_config import get_vocabulary, get_char_to_idx, get_idx_to_char, get_vocab_size, BLANK_TOKEN

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TifinaghOCRInference:
    """Tifinagh OCR inference class"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the OCR model
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._setup_transform()
        self.idx_to_char = get_idx_to_char()
        self.char_to_idx = get_char_to_idx()
        self.blank_idx = self.char_to_idx.get(BLANK_TOKEN, 0)
        
        logger.info(f"TifinaghOCRInference initialized on {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> CRNN:
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model parameters
        if 'config' in checkpoint:
            config = checkpoint['config']
            vocab_size = checkpoint.get('vocab_size', get_vocab_size())
            hidden_size = config.get('hidden_size', 256)
            num_lstm_layers = config.get('num_lstm_layers', 2)
            dropout = config.get('dropout', 0.1)
        else:
            # Fallback to defaults
            vocab_size = checkpoint.get('vocab_size', get_vocab_size())
            hidden_size = checkpoint.get('hidden_size', 256)
            num_lstm_layers = 2
            dropout = 0.1
        
        # Create model
        model = CRNN(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Model parameters: vocab_size={vocab_size}, hidden_size={hidden_size}, num_layers={num_lstm_layers}")
        
        return model
    
    def _setup_transform(self) -> transforms.Compose:
        """Setup image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((32, 128)),  # Standard OCR input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed image tensor
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict(self, image: Union[str, np.ndarray, Image.Image], 
                return_confidence: bool = False) -> Union[str, Tuple[str, float]]:
        """
        Perform OCR on a single image
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predicted text string, or (text, confidence) if return_confidence=True
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            log_probs = self.model(image_tensor)  # [1, sequence_length, vocab_size]
            
            # Decode prediction
            predicted_text = decode_predictions(
                log_probs, 
                self.idx_to_char, 
                blank_idx=self.blank_idx
            )[0]
            
            if return_confidence:
                # Calculate average confidence
                probs = torch.exp(log_probs)  # Convert log probs to probs
                max_probs = torch.max(probs, dim=2)[0]  # [1, sequence_length]
                confidence = torch.mean(max_probs).item()
                return predicted_text, confidence
            else:
                return predicted_text
    
    def predict_batch(self, images: List[Union[str, np.ndarray, Image.Image]], 
                     return_confidence: bool = False) -> List[Union[str, Tuple[str, float]]]:
        """
        Perform OCR on a batch of images
        
        Args:
            images: List of input images
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of predicted text strings or (text, confidence) tuples
        """
        # Preprocess all images
        image_tensors = []
        for image in images:
            image_tensor = self.preprocess_image(image)
            image_tensors.append(image_tensor.squeeze(0))  # Remove batch dim for stacking
        
        # Stack into batch
        batch_tensor = torch.stack(image_tensors).to(self.device)
        
        # Inference
        with torch.no_grad():
            log_probs = self.model(batch_tensor)  # [batch_size, sequence_length, vocab_size]
            
            # Decode predictions
            predicted_texts = decode_predictions(
                log_probs, 
                self.idx_to_char, 
                blank_idx=self.blank_idx
            )
            
            if return_confidence:
                # Calculate confidence for each prediction
                probs = torch.exp(log_probs)  # Convert log probs to probs
                max_probs = torch.max(probs, dim=2)[0]  # [batch_size, sequence_length]
                confidences = torch.mean(max_probs, dim=1).tolist()  # Average over sequence
                
                return list(zip(predicted_texts, confidences))
            else:
                return predicted_texts
    
    def visualize_prediction(self, image: Union[str, np.ndarray, Image.Image], 
                           save_path: str = None, show: bool = True):
        """
        Visualize prediction on image
        
        Args:
            image: Input image
            save_path: Path to save visualization
            show: Whether to display the image
        """
        # Get prediction
        predicted_text, confidence = self.predict(image, return_confidence=True)
        
        # Load original image for visualization
        if isinstance(image, str):
            orig_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            orig_image = Image.fromarray(image if image.dtype == np.uint8 else (image * 255).astype(np.uint8))
        else:
            orig_image = image
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.imshow(orig_image)
        ax.set_title(f"Predicted: '{predicted_text}' (Confidence: {confidence:.3f})", 
                    fontsize=14, pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def evaluate_on_dataset(self, annotations_file: str, images_dir: str) -> dict:
        """
        Evaluate model on a dataset
        
        Args:
            annotations_file: Path to annotations JSON file
            images_dir: Directory containing images
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load annotations
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        correct_predictions = 0
        total_predictions = 0
        character_correct = 0
        character_total = 0
        predictions_list = []
        
        logger.info(f"Evaluating on {len(annotations)} samples...")
        
        # Process in batches for efficiency
        batch_size = 32
        image_paths = []
        ground_truths = []
        
        for image_name, text in annotations.items():
            image_path = os.path.join(images_dir, image_name)
            if os.path.exists(image_path):
                image_paths.append(image_path)
                ground_truths.append(text)
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_gts = ground_truths[i:i+batch_size]
            
            try:
                # Predict batch
                batch_predictions = self.predict_batch(batch_paths)
                
                # Calculate metrics
                for pred, gt in zip(batch_predictions, batch_gts):
                    predictions_list.append({
                        'predicted': pred,
                        'ground_truth': gt,
                        'correct': pred == gt
                    })
                    
                    # Exact match accuracy
                    if pred == gt:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    # Character-level accuracy
                    for pred_char, gt_char in zip(pred, gt):
                        if pred_char == gt_char:
                            character_correct += 1
                        character_total += 1
                    
                    # Account for length differences
                    character_total += abs(len(pred) - len(gt))
                
            except Exception as e:
                logger.error(f"Error processing batch starting at index {i}: {e}")
                total_predictions += len(batch_paths)
        
        # Calculate metrics
        exact_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        char_accuracy = character_correct / character_total if character_total > 0 else 0.0
        
        results = {
            'exact_match_accuracy': exact_accuracy,
            'character_accuracy': char_accuracy,
            'total_samples': total_predictions,
            'correct_predictions': correct_predictions,
            'predictions': predictions_list
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Exact Match Accuracy: {exact_accuracy:.4f}")
        logger.info(f"  Character Accuracy: {char_accuracy:.4f}")
        logger.info(f"  Total Samples: {total_predictions}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Tifinagh OCR Inference")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, help='Path to single image for inference')
    parser.add_argument('--images_dir', type=str, help='Directory containing images for batch inference')
    parser.add_argument('--annotations_file', type=str, help='Annotations file for evaluation')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for inference')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for batch inference')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize OCR model
    ocr = TifinaghOCRInference(args.model_path, args.device)
    
    if args.image_path:
        # Single image inference
        logger.info(f"Processing single image: {args.image_path}")
        
        predicted_text, confidence = ocr.predict(args.image_path, return_confidence=True)
        
        logger.info(f"Predicted text: '{predicted_text}'")
        logger.info(f"Confidence: {confidence:.4f}")
        
        if args.visualize:
            vis_path = output_dir / f"prediction_{Path(args.image_path).stem}.png"
            ocr.visualize_prediction(args.image_path, save_path=str(vis_path), show=False)
    
    elif args.images_dir:
        # Batch inference
        logger.info(f"Processing images in directory: {args.images_dir}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(args.images_dir).glob(f"*{ext}"))
            image_paths.extend(Path(args.images_dir).glob(f"*{ext.upper()}"))
        
        if not image_paths:
            logger.error(f"No images found in {args.images_dir}")
            return
        
        logger.info(f"Found {len(image_paths)} images")
        
        # Process in batches
        all_predictions = []
        for i in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[i:i+args.batch_size]
            batch_predictions = ocr.predict_batch([str(p) for p in batch_paths], return_confidence=True)
            
            for path, (text, conf) in zip(batch_paths, batch_predictions):
                all_predictions.append({
                    'image': path.name,
                    'predicted_text': text,
                    'confidence': conf
                })
        
        # Save results
        results_file = output_dir / 'batch_predictions.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_predictions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Batch predictions saved to {results_file}")
        
        # Print some examples
        logger.info("Sample predictions:")
        for pred in all_predictions[:10]:
            logger.info(f"  {pred['image']}: '{pred['predicted_text']}' (conf: {pred['confidence']:.3f})")
    
    if args.annotations_file and args.images_dir:
        # Evaluation mode
        logger.info("Running evaluation...")
        
        results = ocr.evaluate_on_dataset(args.annotations_file, args.images_dir)
        
        # Save evaluation results
        eval_file = output_dir / 'evaluation_results.json'
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation results saved to {eval_file}")
        
        # Save summary
        summary = {
            'exact_match_accuracy': results['exact_match_accuracy'],
            'character_accuracy': results['character_accuracy'],
            'total_samples': results['total_samples'],
            'correct_predictions': results['correct_predictions']
        }
        
        summary_file = output_dir / 'evaluation_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Show some incorrect predictions for analysis
        incorrect_preds = [p for p in results['predictions'] if not p['correct']]
        if incorrect_preds:
            logger.info(f"\nSample incorrect predictions ({len(incorrect_preds)} total):")
            for pred in incorrect_preds[:10]:
                logger.info(f"  GT: '{pred['ground_truth']}' -> Pred: '{pred['predicted']}'")

if __name__ == "__main__":
    main() 