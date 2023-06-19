import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from prophet import Prophet

bugun=datetime.date.today()
ucay=datetime.timedelta(days=90)
biray=datetime.timedelta(days=30)


tickers=["BTC-USD","THYAO.IS","GOOG","ETH-USD","SOL-USD","GC=F"]

ticker=st.sidebar.text_input("Ticker'ı Giriniz",value="GC=F")

ogrenme=st.sidebar.date_input("Öğrenme Başlangıç Tarihi",value=(bugun-ucay))

tahmin=st.sidebar.date_input("Tahmin Son Günü",value=(bugun+biray))

df=yf.download(ticker,ogrenme,bugun)

df=df.reset_index()

df=df[['Date','Adj Close']]

df.columns=['ds','y']

model=Prophet()
model.fit(df)
gun=(tahmin-bugun).days
gelecek=model.make_future_dataframe(periods=gun)
t=model.predict(gelecek)
st.pyplot(model.plot(t))
