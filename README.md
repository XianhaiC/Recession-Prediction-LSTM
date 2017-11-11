# Recession Prediction LSTM

  This LSTM recurrent neural network learns to predict US economic
  recessions via binary classification with various economic variables fed
  as inputs. Such variables include the S&P 500 index, treasury yield curve
  rates, and the Composite Leading Indicator. The data in the csv files contain 
  data ranging from January 1963 to January 2012 with monthly frequency, 
  spanning a total of 589 months and capturing 7 recessions. The network is 
  implemented with the Keras deep learning
  library.

## Usage

  Before training a model, first format the csv files containing the data
  '''
  data_x, data_y = parse_data()
  '''

  Then create the LSTM model with the specified parameters
  '''
  pred = Prediction(300, 20, 20, 0.2, 3, 12, .5, [0, 1, 6], False, data_x, data_y,)
  '''

  Train the LSTM and have it display the the results afterwards
  '''
  pred.train_LSTM()
  pred.evaluate_LSTM()
  pred.display_results()
  '''
