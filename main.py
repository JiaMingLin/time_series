import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from load_data import load_cla_data

if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


# Define RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outputs, (h_n, _) = self.rnn(x)
        # out = self.fc(h_n.squeeze(0))
        out = self.fc(outputs[:,-1,:])
        out = self.sigmoid(out)
        return out

# Main function
def main():
    # Define stock and date range
    # ticker = 'AAPL'
    # start_date = '2023-01-01'
    # end_date = '2024-01-01'
    sequence_length = 5  # Number of days to consider for each input sequence

    # tra_date = '2014-01-02'
    # val_date = '2015-08-03'
    # tes_date = '2015-10-01'

    # tra_date = '2013-12-31'
    # val_date = '2015-01-07'
    # tes_date = '2015-09-30'

    tra_date = '2019-10-25'
    val_date = '2021-06-04'
    tes_date = '2022-12-30'

    features_train, labels_train, features_val, labels_val, features_test, labels_test = load_cla_data(
        'data/taiex/preprocessed/',
        tra_date, val_date, tes_date
    )

    # Convert data to PyTorch tensors
    features_train = torch.Tensor(features_train)
    labels_train = torch.Tensor(labels_train)
    features_val = torch.Tensor(features_val).to(device)
    labels_val = torch.Tensor(labels_val).to(device)
    features_test = torch.Tensor(features_test).to(device)
    labels_test = torch.Tensor(labels_test).to(device)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    # Create DataLoader
    train_dataset = TensorDataset(features_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=256, num_workers=24, shuffle=True, worker_init_fn=seed_worker, generator=g)

    # Define model, loss function, and optimizer
    input_size = features_train.shape[2]
    hidden_size = 50
    output_size = 1
    model = RNN(input_size, hidden_size, output_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300], gamma=0.5)
    # Train the model
    num_epochs = 400
    
    for epoch in range(num_epochs):
        running_loss = 0
        running_acc = 0
        for inputs, targets in tqdm(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            predicted_labels = (outputs.squeeze() > 0.5).float()
            running_acc += (predicted_labels == targets).float().mean()
        
        scheduler.step()
        running_loss = running_loss/len(train_loader)
        running_acc = running_acc/len(train_loader)
        print(f"Epoch: {epoch:d}, Training loss is {running_loss:.5f}, Training accuracy is {running_acc:.5f}%")
        if (epoch+1) % 3 == 0:
            with torch.no_grad():
                outputs = model(features_val)
                loss = criterion(outputs.squeeze(), labels_val.squeeze())
                predicted_labels = (outputs.squeeze() > 0.5).float()
                accuracy = (predicted_labels == labels_val).float().mean()
                print(f'Epoch: {epoch:d}, Validation loss: {loss.item():.5f}, Accuracy: {accuracy.item()*100:.5f}%')

    # Evaluate the model
    with torch.no_grad():
        outputs = model(features_test)
        predicted_labels = (outputs.squeeze() > 0.5).float()
        accuracy = (predicted_labels == labels_test).float().mean()
        print(f'Testing Accuracy: {accuracy.item()*100:.2f}%')

if __name__ == "__main__":
    main()