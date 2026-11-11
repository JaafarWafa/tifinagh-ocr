#!/usr/bin/env python3
"""
Tifinagh OCR Training Script with CRNN + VGG16-BN
Based on DocTR framework for text recognition training
Supports the data structure from data_generation.py
"""

import os
import json
import argparse
import logging
import datetime
import time
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torchvision import transforms
from torchvision.models import vgg16_bn
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
from tqdm import tqdm
import wandb
from collections import OrderedDict
import cv2

# Import vocabulary configuration
from vocab_config import (
    get_vocabulary, get_char_to_idx, get_idx_to_char, 
    get_vocab_size, BLANK_TOKEN, UNK_TOKEN, PAD_TOKEN
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OCRAugmentations:
    """Custom augmentation classes specifically designed for OCR tasks"""
    
    class RandomPerspective:
        """Apply random perspective transformation"""
        def __init__(self, distortion_scale=0.1, p=0.5):
            self.distortion_scale = distortion_scale
            self.p = p
        
        def __call__(self, img):
            if random.random() > self.p:
                return img
            
            width, height = img.size
            
            # Define perspective transformation points
            startpoints = [[0, 0], [width, 0], [width, height], [0, height]]
            
            # Add random distortion
            endpoints = []
            for point in startpoints:
                x, y = point
                x += random.uniform(-self.distortion_scale * width, self.distortion_scale * width)
                y += random.uniform(-self.distortion_scale * height, self.distortion_scale * height)
                endpoints.append([x, y])
            
            # Apply perspective transformation
            img_array = np.array(img)
            src_points = np.float32(startpoints)
            dst_points = np.float32(endpoints)
            
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            transformed = cv2.warpPerspective(img_array, matrix, (width, height), 
                                            borderMode=cv2.BORDER_REFLECT)
            
            return Image.fromarray(transformed)
    
    class RandomShear:
        """Apply random shear transformation"""
        def __init__(self, shear_range=0.1, p=0.5):
            self.shear_range = shear_range
            self.p = p
        
        def __call__(self, img):
            if random.random() > self.p:
                return img
            
            width, height = img.size
            shear_x = random.uniform(-self.shear_range, self.shear_range)
            shear_y = random.uniform(-self.shear_range, self.shear_range)
            
            # Create affine transformation matrix
            transform_matrix = (1, shear_x, 0, shear_y, 1, 0)
            
            return img.transform(img.size, Image.AFFINE, transform_matrix, 
                               fillcolor=(255, 255, 255))
    
    class RandomElasticTransform:
        """Apply random elastic deformation"""
        def __init__(self, alpha=10, sigma=3, p=0.3):
            self.alpha = alpha
            self.sigma = sigma
            self.p = p
        
        def __call__(self, img):
            if random.random() > self.p:
                return img
            
            img_array = np.array(img)
            shape = img_array.shape[:2]
            
            # Generate random displacement fields
            dx = np.random.uniform(-1, 1, shape) * self.alpha
            dy = np.random.uniform(-1, 1, shape) * self.alpha
            
            # Smooth the displacement fields
            dx = cv2.GaussianBlur(dx, (0, 0), self.sigma)
            dy = cv2.GaussianBlur(dy, (0, 0), self.sigma)
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            x_new = (x + dx).astype(np.float32)
            y_new = (y + dy).astype(np.float32)
            
            # Apply transformation
            if len(img_array.shape) == 3:
                transformed = cv2.remap(img_array, x_new, y_new, cv2.INTER_LINEAR, 
                                      borderMode=cv2.BORDER_REFLECT)
            else:
                transformed = cv2.remap(img_array, x_new, y_new, cv2.INTER_LINEAR, 
                                      borderMode=cv2.BORDER_REFLECT)
            
            return Image.fromarray(transformed)
    
    class RandomNoise:
        """Add various types of noise"""
        def __init__(self, noise_types=['gaussian', 'salt_pepper', 'speckle'], p=0.4):
            self.noise_types = noise_types
            self.p = p
        
        def __call__(self, img):
            if random.random() > self.p:
                return img
            
            img_array = np.array(img).astype(np.float32) / 255.0
            noise_type = random.choice(self.noise_types)
            
            if noise_type == 'gaussian':
                noise = np.random.normal(0, 0.02, img_array.shape)
                img_array = np.clip(img_array + noise, 0, 1)
            
            elif noise_type == 'salt_pepper':
                noise = np.random.random(img_array.shape[:2])
                img_array[noise < 0.01] = 0  # Salt
                img_array[noise > 0.99] = 1  # Pepper
            
            elif noise_type == 'speckle':
                noise = np.random.random(img_array.shape)
                img_array = np.clip(img_array + img_array * noise * 0.1, 0, 1)
            
            return Image.fromarray((img_array * 255).astype(np.uint8))
    
    class RandomBlur:
        """Apply various blur effects"""
        def __init__(self, blur_types=['gaussian', 'motion', 'defocus'], p=0.3):
            self.blur_types = blur_types
            self.p = p
        
        def __call__(self, img):
            if random.random() > self.p:
                return img
            
            blur_type = random.choice(self.blur_types)
            
            if blur_type == 'gaussian':
                radius = random.uniform(0.5, 1.5)
                return img.filter(ImageFilter.GaussianBlur(radius=radius))
            
            elif blur_type == 'motion':
                # Simulate motion blur
                img_array = np.array(img)
                size = random.randint(3, 7)
                angle = random.uniform(0, 360)
                
                # Create motion blur kernel
                kernel = np.zeros((size, size))
                kernel[size//2, :] = 1
                kernel = kernel / size
                
                # Rotate kernel
                center = (size//2, size//2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                kernel = cv2.warpAffine(kernel, rotation_matrix, (size, size))
                
                # Apply blur
                if len(img_array.shape) == 3:
                    blurred = cv2.filter2D(img_array, -1, kernel)
                else:
                    blurred = cv2.filter2D(img_array, -1, kernel)
                
                return Image.fromarray(blurred)
            
            elif blur_type == 'defocus':
                radius = random.uniform(0.5, 2.0)
                return img.filter(ImageFilter.GaussianBlur(radius=radius))
            
            return img
    
    class RandomLighting:
        """Simulate various lighting conditions"""
        def __init__(self, p=0.5):
            self.p = p
        
        def __call__(self, img):
            if random.random() > self.p:
                return img
            
            # Random brightness adjustment
            brightness_factor = random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
            
            # Random contrast adjustment
            contrast_factor = random.uniform(0.8, 1.4)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)
            
            # Random gamma correction
            gamma = random.uniform(0.8, 1.2)
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.power(img_array, gamma)
            img = Image.fromarray((img_array * 255).astype(np.uint8))
            
            return img
    
    class RandomShadow:
        """Add random shadow effects"""
        def __init__(self, p=0.2):
            self.p = p
        
        def __call__(self, img):
            if random.random() > self.p:
                return img
            
            img_array = np.array(img).astype(np.float32)
            height, width = img_array.shape[:2]
            
            # Create random shadow mask
            shadow_intensity = random.uniform(0.3, 0.7)
            
            # Random shadow direction
            if random.random() > 0.5:
                # Vertical shadow
                shadow_width = random.randint(width//4, width//2)
                shadow_start = random.randint(0, width - shadow_width)
                shadow_mask = np.ones((height, width))
                shadow_mask[:, shadow_start:shadow_start + shadow_width] = shadow_intensity
            else:
                # Horizontal shadow
                shadow_height = random.randint(height//4, height//2)
                shadow_start = random.randint(0, height - shadow_height)
                shadow_mask = np.ones((height, width))
                shadow_mask[shadow_start:shadow_start + shadow_height, :] = shadow_intensity
            
            # Apply shadow
            if len(img_array.shape) == 3:
                for c in range(3):
                    img_array[:, :, c] *= shadow_mask
            else:
                img_array *= shadow_mask
            
            return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    class RandomRotationWithPadding:
        """Rotation with intelligent padding to preserve text"""
        def __init__(self, degrees=3, p=0.6):
            self.degrees = degrees
            self.p = p
        
        def __call__(self, img):
            if random.random() > self.p:
                return img
            
            angle = random.uniform(-self.degrees, self.degrees)
            
            # Rotate with white padding
            return img.rotate(angle, fillcolor=(255, 255, 255), expand=False)

class TifinaghDataset(Dataset):
    """Dataset class for Tifinagh OCR with DocTR-style annotations"""
    
    def __init__(self, 
                 annotations_file: str, 
                 images_dir: str, 
                 transform=None, 
                 max_length: int = 32,
                 img_height: int = 32,
                 img_width: int = 128):
        """
        Initialize dataset
        
        Args:
            annotations_file: Path to JSON annotations file
            images_dir: Directory containing images
            transform: Image transformations
            max_length: Maximum text length
            img_height: Target image height
            img_width: Target image width
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.max_length = max_length
        self.img_height = img_height
        self.img_width = img_width
        
        # Vocabulary mappings
        self.char_to_idx = get_char_to_idx()
        self.idx_to_char = get_idx_to_char()
        self.vocab_size = get_vocab_size()
        
        # Special token indices
        self.blank_idx = self.char_to_idx.get(BLANK_TOKEN, 0)
        self.unk_idx = self.char_to_idx.get(UNK_TOKEN, 1)
        self.pad_idx = self.char_to_idx.get(PAD_TOKEN, 2)
        
        # Load annotations
        self.annotations = self._load_annotations(annotations_file)
        
        logger.info(f"Loaded {len(self.annotations)} samples from {annotations_file}")
        logger.info(f"Vocabulary size: {self.vocab_size}")
        logger.info(f"Special tokens - BLANK: {self.blank_idx}, UNK: {self.unk_idx}, PAD: {self.pad_idx}")
    
    def _load_annotations(self, annotations_file: str) -> List[Dict]:
        """Load annotations from JSON file"""
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations_dict = json.load(f)
        
        # Convert to list format
        annotations = []
        for image_name, text in annotations_dict.items():
            annotations.append({
                'image': image_name,
                'text': text
            })
        
        return annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = self.images_dir / annotation['image']
        text = annotation['text']
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Create blank image if loading fails
            image = Image.new('RGB', (self.img_width, self.img_height), color='white')
        
        # Resize image to target size
        image = image.resize((self.img_width, self.img_height), Image.LANCZOS)
        
        if self.transform:
            image = self.transform(image)
        
        # Encode text
        encoded_text = self.encode_text(text)
        text_length = len([idx for idx in encoded_text if idx != self.pad_idx])
        
        return {
            'image': image,
            'text': torch.tensor(encoded_text, dtype=torch.long),
            'text_length': text_length,
            'raw_text': text
        }
    
    def encode_text(self, text: str) -> List[int]:
        """Encode text to indices with padding"""
        encoded = []
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                encoded.append(self.unk_idx)
        
        # Pad or truncate to max_length
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        else:
            encoded.extend([self.pad_idx] * (self.max_length - len(encoded)))
        
        return encoded
    
    def decode_text(self, indices: List[int]) -> str:
        """Decode indices back to text"""
        chars = []
        for idx in indices:
            if idx == self.pad_idx:
                break
            char = self.idx_to_char.get(idx, UNK_TOKEN)
            if char not in [BLANK_TOKEN, UNK_TOKEN, PAD_TOKEN]:
                chars.append(char)
        return ''.join(chars)

class VGG16BNFeatureExtractor(nn.Module):
    """VGG16-BN based feature extractor for CRNN"""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained VGG16-BN
        vgg = vgg16_bn(pretrained=pretrained)
        
        # Extract feature layers (conv layers only)
        self.features = nn.Sequential(*list(vgg.features.children())[:-1])  # Remove last MaxPool
        
        # Add adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # Height=1, preserve width
        
        # Calculate output channels
        self.output_channels = 512  # VGG16 final conv layer
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, 3, height, width]
            
        Returns:
            Feature tensor [batch_size, channels, 1, width]
        """
        features = self.features(x)  # [batch_size, 512, h, w]
        features = self.adaptive_pool(features)  # [batch_size, 512, 1, w]
        return features

class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM layer"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_size]
            
        Returns:
            Output tensor [batch_size, sequence_length, output_size]
        """
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size * 2]
        output = self.linear(lstm_out)  # [batch_size, seq_len, output_size]
        return output

class CRNN(nn.Module):
    """CRNN model with VGG16-BN backbone"""
    
    def __init__(self, 
                 vocab_size: int,
                 hidden_size: int = 128,
                 num_lstm_layers: int = 2,
                 dropout: float = 0.2):
        """
        Initialize CRNN model
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden size for LSTM layers
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Feature extractor (VGG16-BN)
        self.feature_extractor = VGG16BNFeatureExtractor(pretrained=True)
        
        # RNN layers
        self.rnn_layers = nn.ModuleList()
        input_size = self.feature_extractor.output_channels
        
        for i in range(num_lstm_layers):
            output_size = hidden_size if i < num_lstm_layers - 1 else vocab_size
            self.rnn_layers.append(
                BidirectionalLSTM(input_size, hidden_size, output_size)
            )
            input_size = output_size
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, 3, height, width]
            
        Returns:
            Output tensor [batch_size, sequence_length, vocab_size]
        """
        # Extract features
        features = self.feature_extractor(x)  # [batch_size, channels, 1, width]
        
        # Reshape for RNN: [batch_size, width, channels]
        batch_size, channels, height, width = features.size()
        features = features.squeeze(2)  # [batch_size, channels, width]
        features = features.permute(0, 2, 1)  # [batch_size, width, channels]
        
        # Apply RNN layers
        output = features
        for i, rnn_layer in enumerate(self.rnn_layers):
            output = rnn_layer(output)
            if i < len(self.rnn_layers) - 1:  # Don't apply dropout to final layer
                output = self.dropout(output)
        
        # Apply log softmax for CTC loss
        output = F.log_softmax(output, dim=2)
        
        return output

class CTCLoss(nn.Module):
    """CTC Loss wrapper"""
    
    def __init__(self, blank_idx: int = 0, reduction: str = 'mean'):
        super().__init__()
        self.blank_idx = blank_idx
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=True)
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Compute CTC loss
        
        Args:
            log_probs: Log probabilities [batch_size, sequence_length, vocab_size]
            targets: Target sequences [sum(target_lengths)]
            input_lengths: Input sequence lengths [batch_size]
            target_lengths: Target sequence lengths [batch_size]
        """
        # CTC expects [sequence_length, batch_size, vocab_size]
        log_probs = log_probs.permute(1, 0, 2)
        
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    text_lengths = torch.tensor([item['text_length'] for item in batch], dtype=torch.long)
    raw_texts = [item['raw_text'] for item in batch]
    
    # Flatten targets for CTC loss
    targets = []
    for text, length in zip(texts, text_lengths):
        # Remove padding tokens
        valid_text = text[:length]
        targets.extend(valid_text.tolist())
    
    targets = torch.tensor(targets, dtype=torch.long)
    
    # Input lengths (sequence length for each sample)
    input_lengths = torch.full((len(batch),), images.size(3) // 4, dtype=torch.long)  # Approximate after conv
    
    return {
        'images': images,
        'targets': targets,
        'input_lengths': input_lengths,
        'target_lengths': text_lengths,
        'raw_texts': raw_texts
    }

def decode_predictions(log_probs, idx_to_char, blank_idx=0):
    """
    Decode CTC predictions using greedy decoding
    
    Args:
        log_probs: Log probabilities [batch_size, sequence_length, vocab_size]
        idx_to_char: Index to character mapping
        blank_idx: Blank token index
        
    Returns:
        List of decoded strings
    """
    predictions = []
    
    # Get most likely characters
    _, predicted_indices = torch.max(log_probs, dim=2)  # [batch_size, sequence_length]
    
    for sequence in predicted_indices:
        decoded_chars = []
        prev_char = None
        
        for char_idx in sequence:
            char_idx = char_idx.item()
            
            # Skip blank tokens and repeated characters
            if char_idx != blank_idx and char_idx != prev_char:
                char = idx_to_char.get(char_idx, UNK_TOKEN)
                if char not in [BLANK_TOKEN, UNK_TOKEN, PAD_TOKEN]:
                    decoded_chars.append(char)
            
            prev_char = char_idx
        
        predictions.append(''.join(decoded_chars))
    
    return predictions

class TifinaghTrainer:
    """Trainer class for Tifinagh OCR model"""
    
    def __init__(self, 
                 model: CRNN,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 config: Dict):
        """
        Initialize trainer
        
        Args:
            model: CRNN model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Training device
            config: Training configuration
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Store original dataset for progressive augmentation
        self.train_dataset = train_loader.dataset
        
        # Loss function
        self.criterion = CTCLoss(blank_idx=get_char_to_idx().get(BLANK_TOKEN, 0))
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        if config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=config['epochs'],
                eta_min=config['learning_rate'] * 0.01
            )
        elif config['scheduler'] == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config['learning_rate'],
                epochs=config['epochs'],
                steps_per_epoch=len(train_loader)
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        
        # Metrics
        self.best_val_loss = float('inf')
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Character mappings
        self.idx_to_char = get_idx_to_char()
        self.char_to_idx = get_char_to_idx()
        self.blank_idx = self.char_to_idx.get(BLANK_TOKEN, 0)
    
    def update_progressive_augmentation(self, epoch: int):
        """Update augmentation transforms for progressive training"""
        if not self.config.get('progressive_augmentation', False):
            return
        
        # Create new transforms for current epoch
        train_transform, _ = create_progressive_transforms(
            self.config['img_height'], 
            self.config['img_width'], 
            epoch, 
            self.config['epochs']
        )
        
        # Update dataset transform
        self.train_dataset.transform = train_transform
        
        # Log current augmentation level
        progress = epoch / self.config['epochs']
        if progress < 0.3:
            level = 'heavy'
        elif progress < 0.6:
            level = 'medium'
        elif progress < 0.8:
            level = 'light'
        else:
            level = 'light'
        
        logger.info(f"Updated augmentation to {level} level for epoch {epoch+1}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['images'].to(self.device)
            targets = batch['targets'].to(self.device)
            input_lengths = batch['input_lengths'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            log_probs = self.model(images)
            
            # Adjust input lengths based on actual output sequence length
            actual_seq_len = log_probs.size(1)
            input_lengths = torch.clamp(input_lengths, max=actual_seq_len)
            
            # Compute loss
            loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            if self.config['scheduler'] == 'onecycle':
                self.scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'train_loss_step': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['images'].to(self.device)
                targets = batch['targets'].to(self.device)
                input_lengths = batch['input_lengths'].to(self.device)
                target_lengths = batch['target_lengths'].to(self.device)
                raw_texts = batch['raw_texts']
                
                # Forward pass
                log_probs = self.model(images)
                
                # Adjust input lengths
                actual_seq_len = log_probs.size(1)
                input_lengths = torch.clamp(input_lengths, max=actual_seq_len)
                
                # Compute loss
                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
                total_loss += loss.item()
                
                # Decode predictions
                predictions = decode_predictions(log_probs, self.idx_to_char, self.blank_idx)
                
                # Calculate accuracy
                for pred, gt in zip(predictions, raw_texts):
                    if pred == gt:
                        correct_predictions += 1
                    total_predictions += 1
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_accuracy': self.best_accuracy,
            'config': self.config,
            'vocab_size': self.model.vocab_size,
            'hidden_size': self.model.hidden_size
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config['save_dir']) / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config['save_dir']) / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Training for {self.config['epochs']} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config['epochs']):
            # Update progressive augmentation at the start of each epoch
            self.update_progressive_augmentation(epoch)
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_accuracy = self.validate(epoch)
            
            # Update scheduler
            if self.config['scheduler'] == 'plateau':
                self.scheduler.step(val_loss)
            elif self.config['scheduler'] == 'cosine':
                self.scheduler.step()
            
            # Check if best model
            is_best = val_accuracy > self.best_accuracy
            if is_best:
                self.best_accuracy = val_accuracy
                self.best_val_loss = val_loss
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}, "
                f"Best Accuracy: {self.best_accuracy:.4f}"
            )
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'best_accuracy': self.best_accuracy
                })
        
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {self.best_accuracy:.4f}")

def create_data_transforms(img_height: int = 32, img_width: int = 128, augmentation_level: str = 'medium'):
    """
    Create comprehensive data augmentation transforms for OCR
    
    Args:
        img_height: Target image height
        img_width: Target image width
        augmentation_level: 'light', 'medium', 'heavy', or 'extreme'
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    
    # Define augmentation parameters based on level
    aug_params = {
        'light': {
            'perspective_p': 0.2, 'perspective_scale': 0.05,
            'shear_p': 0.2, 'shear_range': 0.05,
            'elastic_p': 0.1, 'elastic_alpha': 5,
            'noise_p': 0.2, 'blur_p': 0.1,
            'lighting_p': 0.3, 'shadow_p': 0.1,
            'rotation_p': 0.4, 'rotation_degrees': 2,
            'color_jitter': (0.1, 0.1, 0.1, 0.05)
        },
        'medium': {
            'perspective_p': 0.4, 'perspective_scale': 0.08,
            'shear_p': 0.3, 'shear_range': 0.08,
            'elastic_p': 0.2, 'elastic_alpha': 8,
            'noise_p': 0.3, 'blur_p': 0.2,
            'lighting_p': 0.4, 'shadow_p': 0.15,
            'rotation_p': 0.5, 'rotation_degrees': 3,
            'color_jitter': (0.15, 0.15, 0.15, 0.08)
        },
        'heavy': {
            'perspective_p': 0.5, 'perspective_scale': 0.12,
            'shear_p': 0.4, 'shear_range': 0.12,
            'elastic_p': 0.3, 'elastic_alpha': 12,
            'noise_p': 0.4, 'blur_p': 0.3,
            'lighting_p': 0.5, 'shadow_p': 0.2,
            'rotation_p': 0.6, 'rotation_degrees': 4,
            'color_jitter': (0.2, 0.2, 0.2, 0.1)
        },
        'extreme': {
            'perspective_p': 0.6, 'perspective_scale': 0.15,
            'shear_p': 0.5, 'shear_range': 0.15,
            'elastic_p': 0.4, 'elastic_alpha': 15,
            'noise_p': 0.5, 'blur_p': 0.4,
            'lighting_p': 0.6, 'shadow_p': 0.25,
            'rotation_p': 0.7, 'rotation_degrees': 5,
            'color_jitter': (0.25, 0.25, 0.25, 0.12)
        }
    }
    
    params = aug_params.get(augmentation_level, aug_params['medium'])
    
    # Training transforms with comprehensive augmentation
    train_transform = transforms.Compose([
        # Geometric augmentations
        OCRAugmentations.RandomPerspective(
            distortion_scale=params['perspective_scale'], 
            p=params['perspective_p']
        ),
        OCRAugmentations.RandomShear(
            shear_range=params['shear_range'], 
            p=params['shear_p']
        ),
        OCRAugmentations.RandomElasticTransform(
            alpha=params['elastic_alpha'], 
            sigma=3, 
            p=params['elastic_p']
        ),
        OCRAugmentations.RandomRotationWithPadding(
            degrees=params['rotation_degrees'], 
            p=params['rotation_p']
        ),
        
        # Photometric augmentations
        transforms.ColorJitter(
            brightness=params['color_jitter'][0],
            contrast=params['color_jitter'][1],
            saturation=params['color_jitter'][2],
            hue=params['color_jitter'][3]
        ),
        OCRAugmentations.RandomLighting(p=params['lighting_p']),
        
        # Noise and blur augmentations
        OCRAugmentations.RandomNoise(p=params['noise_p']),
        OCRAugmentations.RandomBlur(p=params['blur_p']),
        OCRAugmentations.RandomShadow(p=params['shadow_p']),
        
        # Additional standard augmentations
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.1),
        
        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # Random erasing (cutout) - applied after normalization
        transforms.RandomErasing(
            p=0.1, 
            scale=(0.02, 0.1), 
            ratio=(0.3, 3.3), 
            value='random'
        )
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_progressive_transforms(img_height: int = 32, img_width: int = 128, epoch: int = 0, total_epochs: int = 50):
    """
    Create progressive augmentation that starts heavy and reduces over time
    
    Args:
        img_height: Target image height
        img_width: Target image width
        epoch: Current epoch
        total_epochs: Total number of epochs
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Calculate augmentation intensity based on epoch
    progress = epoch / total_epochs
    
    if progress < 0.3:
        level = 'heavy'
    elif progress < 0.6:
        level = 'medium'
    elif progress < 0.8:
        level = 'light'
    else:
        level = 'light'  # Very light augmentation in final epochs
    
    return create_data_transforms(img_height, img_width, level)

def main():
    parser = argparse.ArgumentParser(description="Train Tifinagh OCR with CRNN + VGG16-BN")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--train_split', type=str, default='train', help='Training split name')
    parser.add_argument('--val_split', type=str, default='val', help='Validation split name')
    
    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--num_lstm_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'onecycle', 'plateau'], help='Learning rate scheduler')
    parser.add_argument('--find-lr', action='store_true', help='Gridsearch the optimal LR')
    
    # Image arguments
    parser.add_argument('--img_height', type=int, default=32, help='Image height')
    parser.add_argument('--img_width', type=int, default=128, help='Image width')
    parser.add_argument('--max_text_length', type=int, default=16, help='Maximum text length')
    
    # Data augmentation arguments
    parser.add_argument('--augmentation_level', type=str, default='medium',
                       choices=['light', 'medium', 'heavy', 'extreme'],
                       help='Data augmentation intensity level')
    parser.add_argument('--progressive_augmentation', action='store_true',
                       help='Use progressive augmentation (starts heavy, reduces over time)')
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disable data augmentation for training')
    
    # Training setup
    parser.add_argument('--device', type=str, default=None, help='Training device')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of data loader workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='tifinagh-ocr', help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data paths
    data_dir = Path(args.data_dir)
    train_images_dir = data_dir / 'images' / args.train_split
    val_images_dir = data_dir / 'images' / args.val_split
    train_annotations = train_images_dir / f'{args.train_split}_annotations.json'
    val_annotations = val_images_dir / f'{args.val_split}_annotations.json'
    
    # Verify data paths
    for path in [train_images_dir, val_images_dir, train_annotations, val_annotations]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")
    
    # Create transforms based on augmentation settings
    if args.no_augmentation:
        logger.info("Data augmentation disabled")
        # Use validation transforms for both train and val
        _, val_transform = create_data_transforms(args.img_height, args.img_width, 'light')
        train_transform = val_transform
    elif args.progressive_augmentation:
        logger.info("Using progressive augmentation (will change during training)")
        # Initial transforms (will be updated during training)
        train_transform, val_transform = create_progressive_transforms(
            args.img_height, args.img_width, 0, args.epochs
        )
    else:
        logger.info(f"Using {args.augmentation_level} augmentation level")
        train_transform, val_transform = create_data_transforms(
            args.img_height, args.img_width, args.augmentation_level
        )

    # Create datasets
    train_dataset = TifinaghDataset(
        annotations_file=str(train_annotations),
        images_dir=str(train_images_dir),
        transform=train_transform,
        max_length=args.max_text_length,
        img_height=args.img_height,
        img_width=args.img_width
    )
    
    val_dataset = TifinaghDataset(
        annotations_file=str(val_annotations),
        images_dir=str(val_images_dir),
        transform=val_transform,
        max_length=args.max_text_length,
        img_height=args.img_height,
        img_width=args.img_width
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = CRNN(
        vocab_size=get_vocab_size(),
        hidden_size=args.hidden_size,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'save_dir': str(save_dir),
        'save_every': args.save_every,
        'use_wandb': args.use_wandb,
        'img_height': args.img_height,
        'img_width': args.img_width,
        'max_text_length': args.max_text_length,
        'hidden_size': args.hidden_size,
        'num_lstm_layers': args.num_lstm_layers,
        'dropout': args.dropout,
        'augmentation_level': args.augmentation_level,
        'progressive_augmentation': args.progressive_augmentation,
        'no_augmentation': args.no_augmentation
    }
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config
        )
        wandb.watch(model, log='all', log_freq=100)
    
    # Create trainer and start training
    trainer = TifinaghTrainer(model, train_loader, val_loader, device, config)
    trainer.train()
    
    # Save final model
    
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab_size': get_vocab_size(),
        'hidden_size': args.hidden_size,
        'best_accuracy': trainer.best_accuracy,
        'char_to_idx': get_char_to_idx(),
        'idx_to_char': get_idx_to_char()
    }
    
    final_path = save_dir / 'final_model.pth'
    torch.save(final_checkpoint, final_path)
    logger.info(f"Final model saved to {final_path}")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 