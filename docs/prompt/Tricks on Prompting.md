# Tricks on Prompting

# How does the LLM work?

![截圖 2024-03-11 上午8.08.52.png](./Tricks%20on%20Prompting-assets/截圖%202024-03-11%20上午8.08.52.png)

+ # The tasks we want LLM to do

   - For a stock $s$, given a input sequence of $T$ trading days

      $$
      X=\{x_{t-T+1}, x_{t-T+2}, \ldots,x_t\},
      $$

      where $x_i$ is a vector of quantitative market data, such as prices, volume and etc.

   - Tasks

      - (Regression) Predict the return sequence in the next $T'$ trading days

         $$
         y=\{r_{t+1}, r_{t+2}, \ldots, r_{t+T'}\},
         $$

         where $r_{i} = c_{i}/c_{i-1}$, and $c_i$ is the close price of day $i$

      - (Classification) Predict the direction price change after $T'$ trading days

         - Up or down

+ # Related works

   + ### Predicting the life cycle of product(from TWSC)

      - Fine tuning the Llama-2 with additional training data

         ```!python
         # Step 4: 填入字串
         sys_con = f"You are a data analyst. Each product will have module_category and component_category and sales date. You will judge the life cycle of this product based on module_category, component_category and sales date."
         
         user_con = f"The module_category is {module_category}, component_category is {component_category}, sales date is {sale_date}. Please analyze the life cycle of product."
         
         ans_con = f"The life cycle of product is {during_use} months after sales date."
         
         # Step 5: 存為JSON Line格式
         data = [
             {"role": "system", "content": sys_con},
             {"role": "human", "content": user_con},
             {"role": "assistant", "content": ans_con},
         ]
         ```

      - Testing data

         ```!python
         prompt = f"You are a data analyst. Each product will have module_category and component_category and sales date. You will judge the life cycle of this product based on module_category, component_category and sales date."
         
         user_con = f"The module_category is {module_category}, component_category is {component_category}, sales date is {sale_date}. Please analyze the life cycle of product."
         ```

      - Result

         - Correlation = 0.94

   + ### Comparison with our work

      - Previous work is a regression problem

         - Input is product’s features in a vector

         - Output is product’s life cycle in single scalar

      - Our work is more challenging

         - Input is stock’s historical data in time series(multivariate)

         - Output is stock’s future prices in a series of scalars.

# Questions

- Following the corpus construction from previous work 

   - Our training data

      ```!python
      sys_con = f"You are a quant trader, and predict the future prices of a stock from quantitative market data, including open, high, low, close change, volume, percentage return, within the past {T_past} days"
      
      user_con = f"The open is {open_seq}, high is {high_seq}, low is {low_seq},volume is {volume_seq}, absolute return is {absolute_return_seq} and percentage return is {percentage_return_seq}. Please predict the daily stock prices in next {T_next} days.
      
      ans_con = f'The daily stock prices in the next {T_next} days are {future_percentage_return_seq}'
      ```

   - Our testing data for inferencing

      ```!python
      sys_con = f"You are a quant trader, and predict the future prices of a stock from quantitative market data, including open, high, low, close change, volume, percentage return, within the past {T_past} days"
      
      user_con = f"The open is {open_seq}, high is {high_seq}, low is {low_seq},volume is {volume_seq}, absolute return is {absolute_return_seq} and percentage return is {percentage_return_seq}. Please predict the daily stock prices in next {T_next} days.
      ```

### Question 1: Can the LLM align the multiple time series by date?

- Prompting

   ```!plain
   You are a quant trader, and predict the future prices of a stock from quantitative market data, including close and volume, within the past 5 days. Give me code in python is ok
   ```

- User question

   ```!plain
   The close is {140, 141, 144, 143, 145} and volume is {10000, 12000, 9000, 10010, 11000}. Please predict the daily stock prices in next 5 days
   ```

- Answer by ChatGPT 3.5

   ```!python
   import numpy as np
   import pandas as pd
   
   # Given close prices and volume
   close_prices = np.array([140, 141, 144, 143, 145])
   volume = np.array([10000, 12000, 9000, 10010, 11000])
   
   # Feature engineering
   rolling_mean = pd.Series(close_prices).rolling(window=3).mean()
   
   # Prepare input features for prediction
   X_predict = pd.DataFrame({'Close': close_prices, 'Volume': volume, 'Rolling_mean': rolling_mean})
   ```

- Take a look into training data

   ![Screenshot 2024-03-11 112139.png](./Tricks%20on%20Prompting-assets/Screenshot%202024-03-11%20112139.png)

- So the LLM can perfectly align the time series by date.

### Question 2: Can the LLM fully understand the our task?

- Examining the initial prompt

   ```!plain
   You are a quant trader, and predict the future prices of a stock from quantitative market data, including close and volume, within the past 5 days. Give me code in python is ok
   ```

- Answer by ChatGPT 3.5

   ```python
   import numpy as np
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import mean_squared_error
   import matplotlib.pyplot as plt
   
   # Generate sample data (replace this with your actual data)
   np.random.seed(42)
   dates = pd.date_range('2024-01-01', '2024-01-05')
   close_prices = np.random.rand(len(dates)) * 10 + 50
   volume = np.random.randint(100000, 500000, len(dates))
   
   # Create a DataFrame
   data = {'Date': dates, 'Close': close_prices, 'Volume': volume}
   df = pd.DataFrame(data)
   df.set_index('Date', inplace=True)
   
   # Feature engineering
   df['Rolling_mean'] = df['Close'].rolling(window=3).mean()
   
   # Shift the close price to create the target variable (future close price)
   df['Target'] = df['Close'].shift(-1)
   
   # Drop NaN values introduced by shifting
   df.dropna(inplace=True)
   
   # Split the data into features (X) and target variable (y)
   X = df[['Close', 'Volume', 'Rolling_mean']]
   y = df['Target']
   
   # Split the data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   # Initialize the linear regression model
   model = LinearRegression()
   
   # Train the model
   model.fit(X_train, y_train)
   
   # Make predictions on the test set
   predictions = model.predict(X_test)
   
   # Evaluate the model
   mse = mean_squared_error(y_test, predictions)
   print(f'Mean Squared Error: {mse}')
   
   # Plot actual vs predicted values
   plt.scatter(X_test['Close'], y_test, color='black', label='Actual')
   plt.scatter(X_test['Close'], predictions, color='blue', label='Predicted')
   plt.xlabel('Close Price')
   plt.ylabel('Next Day Close Price')
   plt.legend()
   plt.show()
   ```

- Visualizing the training data

   ![Screenshot 2024-03-11 111307.png](./Tricks%20on%20Prompting-assets/Screenshot%202024-03-11%20111307.png)

- Issue observed

   ```python
   # Initialize the linear regression model
   model = LinearRegression()
   
   # Train the model
   model.fit(X_train, y_train)
   
   # Make predictions on the test set
   predictions = model.predict(X_test)
   ```

   - The LLM regarding the problem as simple regression problem, taking single day’s feature as prediction input.