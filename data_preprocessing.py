import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Feature engineering: Create moving averages and normalize data
def create_features(df):
    df = df.dropna(axis='columns')
    # data['SMA_5'] = data['Close'].rolling(window=5).mean()
    # data['SMA_10'] = data['Close'].rolling(window=10).mean()
    # data = data.dropna()

    df['prev_close'] = df['Close'].shift(1)
    df = df.dropna(axis='rows')

    df['movement_percentage'] = (df['Close']- df['prev_close'])/df['prev_close']
    df['Open'] = (df['Open'] / df['prev_close']) - 1
    df['High'] = (df['High'] / df['prev_close']) - 1
    df['Low'] = (df['Low'] / df['prev_close']) - 1
    df['Close'] = (df['Close'] / df['prev_close']) - 1

    result_df = pd.DataFrame(columns = ['Date','movement_percentage', 'Open', 'High', 'Low', 'Close', 'Volume'])
    result_df['Date'] = df['Date']
    result_df['movement_percentage'] = df['movement_percentage']
    result_df['Open'] = df['Open']
    result_df['High'] = df['High']
    result_df['Low'] = df['Low']
    result_df['Close'] = df['Close']
    result_df['Volume'] = df['Volume']

    scaler = StandardScaler()
    result_df[['Open', 'High', 'Low', 'Close']] = \
    scaler.fit_transform(
        result_df[['Open', 'High', 'Low', 'Close']]
    )

    return result_df

def write_data(dataframe, path):
    dataframe.to_csv(path, index=False)

if __name__ == "__main__":
    raw_data_dir = 'data/kdd17/raw/'
    pre_processed_dir = 'data/kdd17/preprocessed/'
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    stock_list = os.listdir(raw_data_dir)

    for (i, filename) in tqdm(enumerate(stock_list)):
        df_tmp = pd.read_csv(os.path.join(raw_data_dir, filename))
        df_tmp = create_features(df_tmp)
        write_data(df_tmp, os.path.join(pre_processed_dir,filename))
