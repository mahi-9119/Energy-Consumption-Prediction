# Energy Consumption Prediction using Machine Learning  
## Time Series Forecasting (LSTM)  

### Overview  
This project focuses on predicting energy consumption using machine learning, specifically Long Short-Term Memory (LSTM) networks, on data from Finland's transmission system operator. The objective is to explore the potential of machine learning techniques for forecasting electricity consumption, which can be critical for optimizing energy distribution, deploying renewable energy sources, and reducing wastage from polluting standby generation. The model is evaluated using root mean squared error (RMSE), and the results show the effectiveness of machine learning for energy consumption prediction.

### Abstract  
The research aims to predict energy consumption based on historical data from Finland's transmission system operator. The data spans 6 years of hourly electricity consumption in Finland, and it is a univariate time series with seasonal variations. The LSTM model was used to train the dataset, with the goal of testing the performance of machine learning in solving complex forecasting problems. The model's predictions were evaluated using RMSE, which provides a direct comparison to the actual energy consumption data. The successful prediction of electricity consumption using machine learning can help in deploying renewable energy, planning for high/low load days, and minimizing reliance on polluting reserve standby generation.

### Dataset  
The data used in this project was sourced from Finland's transmission system operator and is available as a CSV file. It contains 52,965 observations and 5 variables, with no missing values. The dataset includes the following details:  
- **Total Observations**: 52,965  
- **Number of Variables**: 5  
- **Minimum Load Volume**: 5341 MWh  
- **Maximum Load Volume**: 15,105 MWh  
- **Average Load Volume**: 9488.75 MWh  

The dataset is a univariate time series, where one column represents time, and another column represents the energy consumption. To make predictions at a daily level, the data was down-sampled using the `resample()` function, converting the data from an hourly to a daily frequency.

### Model Implementation  
1. **Data Preprocessing**:  
   The data was first imported as a CSV file and stored in a GitHub repository for version control. Missing values were not present in the dataset, so no data imputation was necessary. The energy consumption data was down-sampled from hourly to daily frequency to predict daily energy consumption.

2. **LSTM Model Training**:  
   The LSTM model was trained using the down-sampled dataset, with a training set and a validation set for evaluation. The model was trained for 60 epochs, and the batch size used during training was 20. The model updated its weights after each batch and was evaluated using RMSE, which measures the difference between predicted and actual energy consumption values.

3. **Evaluation**:  
   The performance of the model was evaluated using the RMSE metric. A lower RMSE value indicates better prediction accuracy. The model's ability to predict electricity consumption accurately is useful for strategic energy planning and reducing reliance on non-renewable energy sources.

### Conclusion  
This project demonstrates the ability of machine learning models, particularly LSTM networks, to forecast energy consumption effectively. The model provides valuable insights for energy distribution, planning for high/low load days, and managing the deployment of renewable energy sources. The use of machine learning in energy forecasting can contribute to reducing energy wastage and improving sustainability in energy systems.

### Requirements  
- Python 3.x  
- TensorFlow  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn

### How to Use  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/yourusername/energy-consumption-prediction.git  
   ```

2. Install the required dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```

3. Load the dataset and preprocess it:
   ```python
   import pandas as pd
   data = pd.read_csv('path/to/dataset.csv')
   ```

4. Train the LSTM model:
   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import LSTM, Dense

   # Build LSTM model
   model = Sequential()
   model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
   model.add(Dense(1))
   model.compile(optimizer='adam', loss='mse')

   # Train model
   model.fit(X_train, y_train, epochs=60, batch_size=20, validation_data=(X_val, y_val))
   ```

5. Evaluate the model's performance:
   ```python
   from sklearn.metrics import mean_squared_error
   predictions = model.predict(X_test)
   rmse = mean_squared_error(y_test, predictions, squared=False)
   print(f'RMSE: {rmse}')
   ```

### License  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
