import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import SOTA_HybridModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def active_learning_patching(data_dir='active_learning_data'):
    patch_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    try:
        active_dataset = datasets.ImageFolder(root=data_dir, transform=patch_transform)
    except FileNotFoundError:
        print("Dataset folder not found!")
        return

    loader = DataLoader(active_dataset, batch_size=4, shuffle=True)
    
    model = SOTA_HybridModel(num_classes=4).to(device)
    model.load_state_dict(torch.load('best_brain_tumor_model_v2.pth', map_location=device))
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    print("\n💉 Injecting knowledge (Fine-tuning)...")
    model.train()
    
    # Freeze BatchNorm
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()
            
    for epoch in range(15):
        total_loss = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    torch.save(model.state_dict(), 'best_brain_tumor_model_patched.pth')
    print("🏆 Patched Model Saved!")

if __name__ == "__main__":
    active_learning_patching()
