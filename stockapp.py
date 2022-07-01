import math 
import pandas_datareader as web 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import streamlit as st 
from keras.models import load_model
from datetime import date


st.title("INVESTRONOMY")

user_input = st.selectbox("SELECT THE STOCK ", ['MSFT','GOOG','TSLA'])
df = web.DataReader(user_input , data_source='stooq' ,start  = '2010-01-01', end = date.today() )

st.subheader("Data from 2010 ")
st.write(df.describe())


st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close[:10])
st.pyplot(fig)



data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing =  pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array= scaler.fit_transform(data_training)
#t transforms data by scaling features to a given range. 
#It scales the values to a specific value range without changing the shape of the original distribution.



model = load_model('keras_model.h5')
#An H5 file is a data file saved in the Hierarchical Data Format (HDF). 
#It contains multidimensional arrays of scientific data. 
#H5 files are commonly used in aerospace, physics, engineering, 
#finance, academic research, genomics, astronomy, electronics instruments, and medical fields


past_100_days=data_training.tail(100)
final_df = past_100_days.append(data_testing)







input_data=scaler.fit_transform(final_df)


x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
             x_test.append(input_data[i-100:i])
             y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

predict=model.predict(x_test)

predict = scaler.inverse_transform(predict)
y_test = np.reshape(y_test, (y_test.shape[0],1))
y_test = scaler.inverse_transform(y_test)



st.subheader("Predicted Price vs Original Price")
fig2 = plt.figure(figsize=(25,16))
plt.plot(y_test[:100],'b',label='Original Price')
plt.plot(predict[:100],'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


