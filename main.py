#!/usr/bin/env python
# coding: utf-8



import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from PIL import Image
import model

START = "2015-01-01"
TODAY = date.today().strftime('%Y-%m-%d')

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    data.Date = data.Date.apply(lambda x: x.strftime('%Y-%m-%d'))
    return data

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y = data['Close'], name = 'Close price'))
    fig.layout.update(title_text = 'Time Series Data', xaxis_rangeslider_visible=True, width = 900, height = 600)
    st.plotly_chart(fig,use_container_width=True)

    
st.title("Stock Predicition App")

st.sidebar.title("Select Dataset, Model and Prediction period")

stocks = ('AAPL', 'GOOG', 'MSFT', 'GME')
selected_stock = st.sidebar.selectbox("Select dataset for prediction", stocks)

data = load_data(selected_stock)

st.subheader('Raw data')
st.write(data.tail())

plot_raw_data()


# Forecasting
models = ('Prophet', 'None')
selected_model = st.sidebar.selectbox("Select model for prediction", models)

n_years = st.sidebar.slider("Years pf prediction:", 1, 4)
period = n_years * 365

st.sidebar.subheader('About')

st.sidebar.markdown("[![Foo](https://i.ibb.co/BTzGtLg/Git-Hub-Mark-32px.png)](https://github.com/Thiefwerty)")

df_train = data[['Date', 'Close']]

model = model.Model(selected_model, df_train, period)
model.start()
