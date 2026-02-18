import torch
from model import HybridCNNTransformerQuantile, LSTMBaseline, QuantileLoss
from data_loader import ERCOTDataset
import matplotlib.pyplot as plt

def train(model, loader, optimizer, criterion, device):
    model.train()
    
    epoch_loss = 0
    for batch in loader:
        x, y, _, _ = batch  # Ignore mean and std from dataset
        x = x.to(device)
        y_max, extreme = y
        
        preds = model(x)
        loss = criterion(preds, y_max)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = ERCOTDataset('Data/Final_dataset_ERCOT_v2.csv')
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    # Models and their optimizers
    models = [
        ('cnn', HybridCNNTransformerQuantile(1).to(device), torch.optim.Adam(HybridCNNTransformerQuantile(1).parameters())),
        ('lstm', LSTMBaseline(24*7, hidden_dim=64).to(device), torch.optim.Adam(LSTMBaseline(24*7, hidden_dim=64).parameters()))
    ]
    
    criterion = QuantileLoss()
    
    epochs = 10  # Train for a fixed number of epochs for the sake of example
    
    train_losses = []
    for epoch in range(epochs):
        for name, model, optimizer in models:
            loss = train(model, loader, optimizer, criterion, device)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss ({name}): {loss:.4f}")
            
            # Save model state dictionary for each epoch
            torch.save(model.state_dict(), f'results/{name}_model_{epoch+1}.pth')
        
        train_losses.append(loss)  # Save the last loss of all models
    
    plt.figure()
    plt.plot(train_losses)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("results/training_curve.png")  # Save training curve plot
    
if __name__ == "__main__":
    main()