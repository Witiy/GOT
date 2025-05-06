import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import copy
from tqdm import tqdm

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, output_dim)
            
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
    @torch.no_grad()
    def transform(self, x):
        x = torch.tensor(x).to(next(self.parameters()).device)
        return self.forward(x).cpu().numpy()

    def save(self, path):
        torch.save(self, path + '/map_model.pt')
        
    
def load_map_model(path):
    model = torch.load(path + '/map_model.pt')
    return model

def learn_embed2vis_map(adata, embedding_key, vis_key, batch_size=256, num_epochs = 100, patience = 5,
                       device=None):
    input_dim = adata.obsm[embedding_key].shape[1]
    output_dim = adata.obsm[vis_key].shape[1]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X = torch.tensor(adata.obsm[embedding_key]).float().to(device)  
    y = torch.tensor(adata.obsm[vis_key]).float().to(device)  

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    
    
    model = SimpleNN(input_dim, output_dim).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    

    best_val_loss = float('inf')
    patience_counter = 0
    
    # training
    
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        model.train()
        train_loss = 0.
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        # 验证模型
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_outputs = model(val_X)
                val_loss += criterion(val_outputs, val_y).item()
        
        val_loss /= len(val_loader)
        pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        #print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # eval
    model.eval()
    with torch.no_grad():
        test_output = model(X)
        test_loss = criterion(test_output, y)
        print(f'Test Loss: {test_loss.item():.4f}')
    model.load_state_dict(best_state)
    
    return model(X).detach().cpu().numpy(), model