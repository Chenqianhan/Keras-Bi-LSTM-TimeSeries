# Keras-Bi-LSTM-TimeSeries
It‘s my graduation design, which is designed to restore missing values in remote-sensing images.

**Actually, the program is mainly divided into three parts: data_cleaning(1), knn classifier(2), Bi-LSTM(3).**  

Because, the CSV file needs restored is mixed with different data types. Thus, (1)we convert these data into float. And by the way the missing elements will be assigned as -999.  

Next, the data arrays are restored one by one. (2)Each input array needs k nearest data arrays to train the Bi-LSTM model. 

(3)The Bi-LSTM model is trained by k nearest arrays. Then we feed the model with input array and get the prediction. The input array inserted with prediction will be input to model again and again. To continuelly  update and optimize the prediction.

