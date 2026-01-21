import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from structure.loaddata_aduc import create_dual_backbone_datasets, dual_backbone_collate_fn
from model import DualBackboneTimeLapseClassifier

def train_dual_backbone_model():
    config = {
        'root_dir': 'D:/stroke/embryo/vin_embryov2/',
        'num_classes': 2,
        'context_size': 5,
        'reference_size': 3,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'sampling_strategy': 'temporal_split',  # or 'uniform', 'random'
    }
    
    print(f"Using device: {config['device']}")
    
    train_dataset, test_dataset = create_dual_backbone_datasets(
        root_dir=config['root_dir'],
        context_size=config['context_size'],
        reference_size=config['reference_size'],
        sampling_strategy=config['sampling_strategy']
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=dual_backbone_collate_fn,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=dual_backbone_collate_fn,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    model = DualBackboneTimeLapseClassifier(
        num_classes=config['num_classes'],
        context_size=config['context_size'],
        reference_size=config['reference_size'],
        context_encoder="efficientnet-b0",
        reference_encoder="efficientnet-b0",
        context_encoding_size=512,
        reference_encoding_size=512,
        mha_num_attention_heads=8,
        mha_num_attention_layers=4,
        dropout_rate=0.1
    ).to(config['device'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]')
        for batch_idx, (context_frames, reference_frames, labels) in enumerate(train_pbar):
            context_frames = context_frames.to(config['device'])
            reference_frames = reference_frames.to(config['device'])
            labels = labels.to(config['device'])
            
            optimizer.zero_grad()
            
            outputs = model(context_frames, reference_frames)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Val]')
            for context_frames, reference_frames, labels in val_pbar:
                context_frames = context_frames.to(config['device'])
                reference_frames = reference_frames.to(config['device'])
                labels = labels.to(config['device'])
                
                outputs = model(context_frames, reference_frames)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        epoch_val_loss = val_loss / len(test_loader)
        epoch_val_acc = 100. * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        scheduler.step(epoch_val_loss)
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_state = model.state_dict().copy()
            print(f"New best validation accuracy: {best_val_acc:.2f}%")
        
        print(f'Epoch {epoch+1}/{config["num_epochs"]}:')
        print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 60)
        
        if (epoch + 1) % 10 == 0:
            print("\nDetailed Classification Report:")
            print(classification_report(all_labels, all_predictions, 
                                      target_names=['Normal', 'Abnormal']))
            print("\nConfusion Matrix:")
            print(confusion_matrix(all_labels, all_predictions))
            print('-' * 60)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation accuracy: {best_val_acc:.2f}%")

    model.eval()
    final_predictions = []
    final_labels = []
    final_probabilities = []
    
    with torch.no_grad():
        for context_frames, reference_frames, labels in tqdm(test_loader, desc='Final Evaluation'):
            context_frames = context_frames.to(config['device'])
            reference_frames = reference_frames.to(config['device'])
            
            probs = model.predict(context_frames, reference_frames)
            predictions = torch.argmax(probs, dim=1)
            
            final_predictions.extend(predictions.cpu().numpy())
            final_labels.extend(labels.numpy())
            final_probabilities.extend(probs.cpu().numpy())
    
    final_accuracy = accuracy_score(final_labels, final_predictions)
    print(f"\nFinal Test Accuracy: {final_accuracy*100:.2f}%")
    print("\nFinal Classification Report:")
    print(classification_report(final_labels, final_predictions, 
                              target_names=['Normal', 'Abnormal']))
    print("\nFinal Confusion Matrix:")
    print(confusion_matrix(final_labels, final_predictions))
    
    torch.save({
        'model_state_dict': best_model_state,
        'config': config,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'final_test_acc': final_accuracy
    }, 'dual_backbone_best_model.pth')
    
    print("\nModel saved as 'dual_backbone_best_model.pth'")
    
    return model, {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'final_test_acc': final_accuracy
    }


def load_and_evaluate_model(model_path, test_dataset):
    checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    config = checkpoint['config']
    
    model = DualBackboneTimeLapseClassifier(
        num_classes=config['num_classes'],
        context_size=config['context_size'],
        reference_size=config['reference_size']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model with best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    print(f"Final test accuracy from training: {checkpoint['final_test_acc']*100:.2f}%")
    
    return model


if __name__ == "__main__":
    model, history = train_dual_backbone_model()

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracies'], label='Train Acc')
    plt.plot(history['val_accuracies'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()