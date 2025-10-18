#!/usr/bin/env python3
"""
ResNet-50 Binary Classifier for Welding Anomaly Detection
Fine-tuned ResNet-50 for binary classification of weld ROIs (good/bad)
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, f1_score, average_precision_score
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
from PIL import Image, ImageEnhance
import copy


class LetterboxResize:
    """Enhanced letterbox resize for optimal quality"""
    
    def __init__(self, target_size=(224, 672)):
        self.target_size = target_size
    
    def __call__(self, image):
        """Single high-quality resize operation"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scale to fit in target size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Single high-quality resize with LANCZOS (best for downscaling)
        if scale < 1.0:  # Downscaling
            interpolation = cv2.INTER_AREA  # Best for downscaling
        else:  # Upscaling
            interpolation = cv2.INTER_CUBIC  # Best for upscaling
            
        resized = cv2.resize(img_array, (new_w, new_h), interpolation=interpolation)
        
        # Create canvas and center image
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return Image.fromarray(canvas)

class WeldAugmentation:
    """Light augmentations for weld ROIs"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image):
        """Apply random augmentations"""
        if random.random() < self.p:
            # Horizontal flip (if orientation doesn't matter for your welds)
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Brightness and contrast
            if random.random() < 0.5:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(random.uniform(0.8, 1.2))
            
            if random.random() < 0.5:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(random.uniform(0.8, 1.2))
            
            # Sharpness
            if random.random() < 0.3:
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(random.uniform(0.8, 1.5))
            
            # Small rotation (±5 degrees)
            if random.random() < 0.3:
                angle = random.uniform(-5, 5)
                image = image.rotate(angle, fillcolor=(114, 114, 114))
        
        return image

class WeldingDataset(Dataset):
    """Dataset for welding patch images with binary classification"""
    
    def __init__(self, data_dir, transform=None, class_type='both', is_training=False):
        """
        Initialize dataset
        
        Args:
            data_dir (str): Directory containing welding patches
            transform: Torchvision transforms
            class_type (str): 'good', 'bad', or 'both'
            is_training (bool): Whether this is training data (for augmentation)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_type = class_type
        self.is_training = is_training
        
        self.image_paths = []
        self.labels = []
        
        if class_type == 'both':
            # Load both good and bad images
            for class_name, label in [('good', 0), ('bad', 1)]:
                class_dir = self.data_dir / class_name
                if class_dir.exists():
                    paths = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
                    self.image_paths.extend(paths)
                    self.labels.extend([label] * len(paths))
                    print(f"Found {len(paths)} images in {class_dir}")
        else:
            # Load single class
            class_dir = self.data_dir / class_type
            if not class_dir.exists():
                raise ValueError(f"Directory {class_dir} does not exist")
                
            self.image_paths = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            label = 0 if class_type == 'good' else 1
            self.labels = [label] * len(self.image_paths)
            print(f"Found {len(self.image_paths)} images in {class_dir}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.float32),
            'image_path': str(img_path)
        }

class WeldResNet(nn.Module):
    """ResNet-50 binary classifier for weld ROI classification"""
    
    def __init__(self, pretrained=True, freeze_stages=2):
        """
        Initialize ResNet-50 classifier
        
        Args:
            pretrained (bool): Use ImageNet pretrained weights
            freeze_stages (int): Number of initial stages to freeze (0-4)
        """
        super(WeldResNet, self).__init__()
        
        # Load pretrained ResNet-50
        self.backbone = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        # self.backbone = models.resnet50(weights='IMAGENET1K_V2')
        # total params
        total_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"Total parameters: {total_params:,}")

        # trainable params
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"Trainable parameters before freezing: {trainable_params:,}")
        # Replace the classifier head
        in_feats = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_feats, 1)
        
        # Freeze initial stages if specified
        if freeze_stages > 0:
            self.freeze_stages(freeze_stages)
    
    def freeze_stages(self, num_stages):
        """Freeze the first num_stages of the network"""
        stages = [
            self.backbone.features[0],  # stem
            self.backbone.features[1],  # stage 1
            self.backbone.features[2],  # stage 2
            self.backbone.features[3],  # stage 3
            self.backbone.features[5],  # stage 4
            self.backbone.features[6],  # stage 4
            self.backbone.features[7],  # stage 4
        ]
        
        for i in range(min(num_stages, len(stages))):
            for module_list in stages[i]:
                if isinstance(module_list, nn.Module):
                    for param in module_list.parameters():
                        param.requires_grad = False
                else:
                    # It's a list of modules
                    for module in module_list:
                        for param in module.parameters():
                            param.requires_grad = False
        # trainable params
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"Trainable parameters after freezing: {trainable_params:,}")
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x).squeeze(1)  # Remove extra dimension for BCEWithLogitsLoss

class WeldingClassifier:
    """ResNet-50 based welding binary classifier"""
    
    def __init__(self, train_data_dir, val_data_dir=None, test_data_dir=None, device=None, 
                 target_size=(224, 672), short_side=224):
        """
        Initialize welding classifier
        
        Args:
            train_data_dir (str): Directory containing training welding patches
            val_data_dir (str): Directory containing validation welding patches
            test_data_dir (str): Directory containing test welding patches
            device: Torch device
            target_size (tuple): Target canvas size (height, width)
            short_side (int): Target short side length
        """
        self.train_data_dir = Path(train_data_dir)
        self.val_data_dir = Path(val_data_dir) if val_data_dir else None
        self.test_data_dir = Path(test_data_dir) if test_data_dir else None
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.target_size = target_size
        self.short_side = short_side
        
        # Data transforms
        self.train_transform = transforms.Compose([
            LetterboxResize(target_size=target_size),
            WeldAugmentation(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            LetterboxResize(target_size=target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Model and training components
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_threshold = 0.5
        self.best_f1 = 0.0
        
    def build_dataloaders(self, batch_size=32):
        """Build train, validation, and test dataloaders"""
        dataloaders = {}
        
        # Training dataloader
        train_dataset = WeldingDataset(
            self.train_data_dir,
            transform=self.train_transform,
            class_type='both',
            is_training=True
        )
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Validation dataloader
        if self.val_data_dir and self.val_data_dir.exists():
            val_dataset = WeldingDataset(
                self.val_data_dir,
                transform=self.val_transform,
                class_type='both',
                is_training=False
            )
            dataloaders['val'] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Test dataloader
        if self.test_data_dir and self.test_data_dir.exists():
            test_dataset = WeldingDataset(
                self.test_data_dir,
                transform=self.val_transform,
                class_type='both',
                is_training=False
            )
            dataloaders['test'] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        return dataloaders
    
    def initialize_model(self, freeze_stages=5):
        """Initialize the ResNet-50 model"""
        self.model = WeldResNet(pretrained=True, freeze_stages=freeze_stages).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters")
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(dataloader, desc="Training"):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # For metrics
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = running_loss / len(dataloader)
        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
        
        return avg_loss, auc
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                # For metrics
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = running_loss / len(dataloader)
        
        # Calculate metrics
        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_preds)
            pr_auc = average_precision_score(all_labels, all_preds)
            
            # Find optimal threshold
            precisions, recalls, thresholds = precision_recall_curve(all_labels, all_preds)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_f1 = f1_scores[best_idx]
        else:
            auc = pr_auc = best_f1 = 0.0
            best_threshold = 0.5
        
        return avg_loss, auc, pr_auc, best_f1, best_threshold
    
    def train(self, epochs=30, batch_size=32, patience=5, save_dir='./checkpoints'):
        """
        Train the model with early stopping
        
        Args:
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size
            patience (int): Early stopping patience
            save_dir (str): Directory to save checkpoints
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Build dataloaders
        dataloaders = self.build_dataloaders(batch_size)
        
        if 'train' not in dataloaders:
            raise ValueError("Training data not found")
        
        # Initialize model
        self.initialize_model()
        
        # Training history
        history = {
            'train_loss': [], 'train_auc': [],
            'val_loss': [], 'val_auc': [], 'val_pr_auc': [], 'val_f1': []
        }
        
        best_val_f1 = 0.0
        patience_counter = 0
        best_model_state = None
        
        print(f"\n=== Starting Training ===")
        print(f"Epochs: {epochs}, Batch Size: {batch_size}, Patience: {patience}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_auc = self.train_epoch(dataloaders['train'])
            history['train_loss'].append(train_loss)
            history['train_auc'].append(train_auc)
            
            print(f"Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
            
            # Validation
            if 'val' in dataloaders:
                val_loss, val_auc, val_pr_auc, val_f1, val_threshold = self.validate(dataloaders['val'])
                history['val_loss'].append(val_loss)
                history['val_auc'].append(val_auc)
                history['val_pr_auc'].append(val_pr_auc)
                history['val_f1'].append(val_f1)
                
                print(f"Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, PR-AUC: {val_pr_auc:.4f}, F1: {val_f1:.4f}")
                
                # Early stopping and model saving
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    self.best_threshold = val_threshold
                    self.best_f1 = val_f1
                    patience_counter = 0
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    
                    # Save best model
                    torch.save({
                        'model_state_dict': best_model_state,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'best_f1': best_val_f1,
                        'best_threshold': self.best_threshold,
                        'history': history
                    }, save_path / 'best_model.pth')
                    
                    print(f"New best F1: {best_val_f1:.4f} (threshold: {self.best_threshold:.3f})")
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Unfreeze after a few epochs for fine-tuning
            if epoch == 5:
                print("Unfreezing all layers for fine-tuning...")
                self.model.unfreeze_all()
                # Reduce learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nLoaded best model with F1: {self.best_f1:.4f}")
        
        # Save final model and history
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'best_threshold': self.best_threshold,
            'history': history
        }, save_path / 'final_model.pth')
        
        # Plot training history
        self.plot_training_history(history, save_path)
        
        return history
    
    def plot_training_history(self, history, save_dir):
        """Plot and save training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history and history['val_loss']:
            axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # AUC
        axes[0, 1].plot(history['train_auc'], label='Train AUC')
        if 'val_auc' in history and history['val_auc']:
            axes[0, 1].plot(history['val_auc'], label='Val AUC')
        axes[0, 1].set_title('AUC-ROC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # PR-AUC
        if 'val_pr_auc' in history and history['val_pr_auc']:
            axes[1, 0].plot(history['val_pr_auc'], label='Val PR-AUC')
            axes[1, 0].set_title('Precision-Recall AUC')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('PR-AUC')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # F1 Score
        if 'val_f1' in history and history['val_f1']:
            axes[1, 1].plot(history['val_f1'], label='Val F1')
            axes[1, 1].set_title('F1 Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate(self, test_data_dir=None, save_dir=None):
        """
        Evaluate model on test data
        
        Args:
            test_data_dir (str): Directory containing test images (if not provided, uses self.test_data_dir)
            save_dir (str): Directory to save results
        """
        if self.model is None:
            raise ValueError("Model not initialized. Train the model first or load a checkpoint.")
        
        # Use provided test_data_dir or default
        eval_data_dir = test_data_dir if test_data_dir else self.test_data_dir
        if not eval_data_dir or not Path(eval_data_dir).exists():
            raise ValueError(f"Test data directory not found: {eval_data_dir}")
        
        # Build test dataloader
        test_dataset = WeldingDataset(
            eval_data_dir,
            transform=self.val_transform,
            class_type='both',
            is_training=False
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print("Running evaluation on test data...")
        
        self.model.eval()
        all_probs = []
        all_labels = []
        all_paths = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                labels = batch['label']
                paths = batch['image_path']
                
                outputs = self.model(images)
                probs = torch.sigmoid(outputs)
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_paths.extend(paths)
        
        # Convert to numpy arrays
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Make predictions using best threshold
        predictions = (all_probs > self.best_threshold).astype(int)
        
        # Calculate metrics
        auc_score = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
        f1 = f1_score(all_labels, predictions)
        
        print(f"\n=== Evaluation Results ===")
        print(f"AUC-ROC Score: {auc_score:.4f}")
        print(f"PR-AUC Score: {pr_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Threshold used: {self.best_threshold:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, predictions, target_names=['Good', 'Bad']))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, predictions)
        print("\nConfusion Matrix:")
        print("Predicted:  Good  Bad")
        print(f"Good:       {cm[0,0]:4d}  {cm[0,1]:3d}")
        print(f"Bad:        {cm[1,0]:4d}  {cm[1,1]:3d}")
        
        # Save results
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            with open(save_path / 'test_metrics.txt', 'w') as f:
                f.write(f"AUC-ROC Score: {auc_score:.4f}\n")
                f.write(f"PR-AUC Score: {pr_auc:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")
                f.write(f"Threshold: {self.best_threshold:.4f}\n")
                f.write(f"Confusion Matrix:\n")
                f.write(f"True\\Pred  Good  Bad\n")
                f.write(f"Good      {cm[0,0]:4d}  {cm[0,1]:3d}\n")
                f.write(f"Bad       {cm[1,0]:4d}  {cm[1,1]:3d}\n")
            
            # Plot results
            self.plot_evaluation_results(all_labels, all_probs, predictions, save_path)
            
            print(f"Results saved to: {save_path}")
        
        return {
            'auc_roc': auc_score,
            'pr_auc': pr_auc,
            'f1': f1,
            'threshold': self.best_threshold,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': all_probs,
            'labels': all_labels,
            'paths': all_paths
        }
    
    def plot_evaluation_results(self, labels, probs, predictions, save_dir):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Score distribution
        good_scores = probs[labels == 0]
        bad_scores = probs[labels == 1]
        
        axes[0, 0].hist(good_scores, bins=30, alpha=0.7, label='Good Welds', color='green', density=True)
        axes[0, 0].hist(bad_scores, bins=30, alpha=0.7, label='Bad Welds', color='red', density=True)
        axes[0, 0].axvline(self.best_threshold, color='black', linestyle='--', 
                          label=f'Threshold: {self.best_threshold:.3f}')
        axes[0, 0].set_xlabel('Probability')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Distribution of Predictions')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_score = roc_auc_score(labels, probs)
        
        axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precisions, recalls, _ = precision_recall_curve(labels, probs)
        pr_auc = average_precision_score(labels, probs)
        
        axes[1, 0].plot(recalls, precisions, label=f'PR Curve (AUC = {pr_auc:.3f})')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confusion Matrix
        cm = confusion_matrix(labels, predictions)
        im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1, 1].set_title('Confusion Matrix')
        tick_marks = np.arange(2)
        axes[1, 1].set_xticks(tick_marks)
        axes[1, 1].set_yticks(tick_marks)
        axes[1, 1].set_xticklabels(['Good', 'Bad'])
        axes[1, 1].set_yticklabels(['Good', 'Bad'])
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                           horizontalalignment="center",
                           color="white" if cm[i, j] > thresh else "black")
        
        axes[1, 1].set_ylabel('True Label')
        axes[1, 1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model if not already done
        if self.model is None:
            self.initialize_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_threshold = checkpoint.get('best_threshold', 0.5)
        self.best_f1 = checkpoint.get('best_f1', 0.0)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Best F1: {self.best_f1:.4f}, Threshold: {self.best_threshold:.4f}")
        
        return checkpoint.get('history', {})
