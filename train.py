import torch
import torch.nn as nn
import torch.optim as optim
import copy
from model import SOTA_HybridModel
from dataset import load_data

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10

def train_model():
    train_loader, test_loader, classes = load_data(BATCH_SIZE)
    model = SOTA_HybridModel(num_classes=len(classes)).to(device)
    
    # Weighted Loss
    class_weights = torch.tensor([2.0, 1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print("🚀 Starting Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation Loop (Simplified for brevity)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        epoch_val_loss = val_loss / len(test_loader)
        scheduler.step(epoch_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss {running_loss/len(train_loader):.4f}, Val Loss {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'best_brain_tumor_model_v2.pth')
            print("  -> 🔥 New Best Model Saved!")

if __name__ == "__main__":
    train_model()
