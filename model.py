#!/usr/bin/env python
# coding: utf-8

from prophet import Prophet
import streamlit as st
from prophet.plot import plot_plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import graph_objs as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
import tensorflow as tf 



@st.cache
def prophet_data(data, period):

    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    return model, future, forecast


@st.cache
def prophet_validation_data(data):

    data = data[:-365]
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    return model, future, forecast

def prophet_show_forecast(df, num_predictions, num_values):
    # верхняя граница доверительного интервала прогноза
    upper_bound = go.Scatter(
        name='Upper Bound',
        x=df.tail(num_predictions).index,
        y=df.tail(num_predictions).yhat_upper,
        mode='lines',
        marker=dict(color="#444444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')

    # прогноз
    forecast = go.Scatter(
        name='Prediction',
        x=df.tail(num_predictions).index,
        y=df.tail(num_predictions).yhat,
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
    )

    # нижняя граница доверительного интервала
    lower_bound = go.Scatter(
        name='Lower Bound',
        x=df.tail(num_predictions).index,
        y=df.tail(num_predictions).yhat_lower,
        marker=dict(color="#444444"),
        line=dict(width=0),
        mode='lines')

    # фактические значения
    fact = go.Scatter(
        name='Fact',
        x=df.tail(num_values).index,
        y=df.tail(num_values).y,
        marker=dict(color="red"),
        mode='lines',
    )

    # последовательность рядов в данном случае важна из-за применения заливки
    data = [lower_bound, upper_bound, forecast, fact]

    layout = go.Layout(
        yaxis=dict(title='Price'),
        title='Time Series Data',
        showlegend = False)

    fig = go.Figure(data=data, layout=layout)
    
    return fig

def start_prophet(data, period):
    
    data = data.rename(columns= {'Date': 'ds', 'Close':'y'})

    model, future, forecast = prophet_data(data, period)
    val_model, val_future, val_forecast = prophet_validation_data(data)

    st.subheader('Forecast Data')
    st.write(forecast.tail())

    st.write('Forecast Data')
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1, use_container_width=True)

    st.write('Forecast Components')
    fig2 = model.plot_components(forecast)
    st.write(fig2, use_container_width=True)

    st.subheader('Forecast Validation Data')
    st.write(val_forecast.tail())

    st.subheader('Forecast Quality')
    st.write('Calculated based on data from the last year')
    
    val_df = val_forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(data.set_index('ds'))

    val_df['e'] = val_df['y'] - val_df['yhat']
    val_df['p'] = 100 * val_df['e'] /val_df['y']
    st.write(f'MAPE: {np.mean(abs(val_df[-365:].p))}%')
    st.write(f'MAE: {np.mean(abs(val_df[-365:].e))}')
    
    st.subheader('Forecast Quality Visualization')
    st.write(prophet_show_forecast(val_df, 356, 1000), use_container_width=True)


    
class Model():
    
    def __init__(self, model_type, data, period):
        
        self.model_type = model_type
        self.data = data
        self.period = period
        
    def start(self):
        
        if self.model_type == 'Prophet':
            start_prophet(self.data, self.period)
