import vnstock as vns
import pandas as pd
import streamlit as st
import matplotlib as plt
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
Start = '2014-05-20'
Today = date.today().strftime("%Y-%m-%d")

st.title('Stock Price Prediction Model')

Companies = ('FPT','APC', 'ING', 'HDS', 'FUEKIVND')
selected_company = st.selectbox('Select dataset for prediction', Companies)

n_years = st.slider('Years of prediction:',1 ,10)
period = n_years*365



#data = pd.DataFrame(fpt).to_dict(orient='records')
@st.cache_data
def load_data(sticker):
    data = vns.stock_historical_data(symbol=sticker, start_date=Start, end_date=Today, resolution='1D', type='stock')
    return data

data_load_state = st.text('Load data...')
data = load_data(selected_company)
data_load_state.text('Loading data done!')

st.subheader('Raw data')
st.write(data)

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['time'], y=data['open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['time'], y=data['close'], name='stock_close'))
    fig.layout.update(title_text ='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting
df_train = data[['time', 'close']]
df_train = df_train.rename(columns ={'time': 'ds', 'close':'y'})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast)

st.subheader('Forecast data graph')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)