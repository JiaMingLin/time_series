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

# Download historical stock data
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Feature engineering: Create moving averages and normalize data
def create_features(data):
    data = data.dropna(axis='columns')
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data = data.dropna()

    scaler = StandardScaler()
    data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_10']] = \
    scaler.fit_transform(
        data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_10']]
    )
    
    return data

# Label generation: Create binary labels based on price movement
def create_labels(data):
    data['Price_Up'] = (data['Close'] < data['Close'].shift(-1)).astype(int)
    return data

# Prepare data for RNN
def prepare_rnn_data(data, sequence_length):
    features = []
    labels = []

    for i in range(len(data) - sequence_length):
        sequence = data.iloc[i:i+sequence_length][['Close', 'SMA_5', 'SMA_10', 'Volume']].values
        label = data.iloc[i+sequence_length]['Price_Up']
        features.append(sequence)
        labels.append(label)

    return np.array(features), np.array(labels)

def read_data(ticker):
    df = pd.read_csv(os.path.join("trading-with-deep-nn", DATA_DIR, DATASET, 'raw', ticker+".csv"))
    return df

# Define RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.rnn(x)
        out = self.fc(h_n.squeeze(0))
        out = self.sigmoid(out)
        return out

# Main function
def main():
    # Define stock and date range
    # ticker = 'AAPL'
    # start_date = '2023-01-01'
    # end_date = '2024-01-01'
    sequence_length = 5  # Number of days to consider for each input sequence

    tra_date = '2014-01-02'
    val_date = '2015-08-03'
    tes_date = '2015-10-01'

    # # Download stock data
    # stock_data = download_stock_data(ticker, start_date, end_date)

    # # Feature engineering
    # stock_data = create_features(stock_data)

    # # Label generation
    # stock_data = create_labels(stock_data)

    # # Prepare data for RNN
    # features, labels = prepare_rnn_data(stock_data, sequence_length)

    features_train, labels_train, features_val, labels_val, features_test, labels_test = load_cla_data(
        'data/kdd17/preprocessed/',
        tra_date, val_date, tes_date
    )

    # Split data into training and testing sets
    # features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    features_train = torch.Tensor(features_train)
    labels_train = torch.Tensor(labels_train)
    features_val = torch.Tensor(features_val)
    labels_val = torch.Tensor(labels_val)

    # Create DataLoader
    train_dataset = TensorDataset(features_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define model, loss function, and optimizer
    input_size = features_train.shape[2]
    hidden_size = 50
    output_size = 1
    model = RNN(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        for inputs, targets in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 3 == 0:
            with torch.no_grad():
                output = model(features_val)
                predicted_labels = (outputs.squeeze() > 0.5).float()
                accuracy = (predicted_labels == labels_val).float().mean()
                print(f'Epoch: {epoch:d}, Validation Accuracy: {accuracy.item()*100:.2f}%')

    # Evaluate the model
    with torch.no_grad():
        outputs = model(features_test)
        predicted_labels = (outputs.squeeze() > 0.5).float()
        accuracy = (predicted_labels == labels_test).float().mean()
        print(f'Testing Accuracy: {accuracy.item()*100:.2f}%')

if __name__ == "__main__":
    main()