#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pystan==2.19.1.1 --quiet')
get_ipython().system('pip install prophet --quiet')
get_ipython().system('pip install yfinance --quiet')


# In[2]:


import pandas as pd
import yfinance as yf
from datetime import datetime
from datetime import timedelta
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import warnings

warnings.filterwarnings('ignore')

pd.options.display.float_format = '${:,.2f}'.format


# In[3]:


today = datetime.today().strftime('%Y-%m-%d')
start_date = '2016-01-01'

eth_df = yf.download('ETH-USD',start_date, today)

eth_df.tail() 


# In[4]:


eth_df.info()


# In[5]:


eth_df.isnull().sum()


# In[6]:


eth_df.columns


# In[7]:


eth_df.reset_index(inplace=True)
eth_df.columns


# In[8]:


df = eth_df[["Date", "Open"]]

new_names = {
    "Date": "ds", 
    "Open": "y",
}

df.rename(columns=new_names, inplace=True)


# In[9]:


# plot the open price

x = df["ds"]
y = df["y"]

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y))

# Set title
fig.update_layout(
    title_text="Time series plot of Ethereum Open Price",
)

fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
        rangeslider=dict(visible=True),
        type="date",
    )
)


# In[10]:


m = Prophet(
    seasonality_mode="multiplicative" 
)

m.fit(df)


# In[11]:


future = m.make_future_dataframe(periods = 365)
future.tail()


# In[12]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[13]:


next_day = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

forecast[forecast['ds'] == next_day]['yhat'].item()


# In[14]:


plot_plotly(m, forecast)


# In[15]:


plot_components_plotly(m, forecast)


# In[ ]:




