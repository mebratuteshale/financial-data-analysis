import yfinance as yf
import talib as ta
import pandas as pd
import numpy as np
import plotly.express as px
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt


class FinancialAnalyzer:
    def __init__(self,ticker,start_date,end_date):
        self.ticker=ticker
        self.start_date=start_date
        self.end_date=end_date
    def retrieve_stock_data(self):
        return yf.download(self.ticker,start=self.start_date,end=self.end_date)
    def calculate_moving_average(self,data,window_size):
        return ta.SMA(data,timeperiod=window_size)
    def calculate_technical_indicators(self,data):
        # calculate various techical indicators
        data['SMA']=self.calculate_moving_average(data['Close'],20)
        data['RSI']=ta.RSI(data['Close'],timeperiod=14)
        data['EMA']=ta.EMA(data['Close'],timeperiod=20)
        macd,macd_signal,_=ta.MACD(data['Close'])
        data['MACD']=macd 
        data['MACD_Signal']=macd_signal
        # Add more indicators as needed
        return data
    def plot_stock_data(self,data):
        fig = px.line(data,x=data.index,y=['Close','SMA'],title='Stock Price with Moving')
        fig.show()
    def plot_rsi(self,data):
        fig=px.line(data,x=data.index,y='RSI',title='Relative Strength Index (RSI)')
        fig.show
    def plot_ema(self,data):
        fig=px.line(data,x=data.index,y=['Close','EMA'],title='Stock Price with Exponent')
        fig.show()
    def plot_macd(self,data):
        fig=px.line(data,x=data.index,y=['MACD','MACD_Signal'],title='Moving Average ')
    def calculate_portfolio_weight(self,tickers,start_date,end_date):
        data=yf.download(tickers,start=start_date,end=end_date)['Close']
        mu=expected_returns.mean_historical_return(data)
        cov=risk_models.sample_cov(data)
        ef=EfficientFrontier(mu,cov)
        weights=ef.max_sharpe()
        weights=dict(zip(tickers,weights.values()))
        return weights
    def calculate_portfolio_performance(self,tickers,start_date,end_date):
        data=yf.download(tickers,start=start_date,end=end_date)['Close']
        mu=expected_returns.mean_historical_return(data)
        cov=risk_models.sample_cov(data)
        ef=EfficientFrontier(mu,cov)
        weights=ef.max_sharpe()
        portfolio_return, portfolio_volatility, sharpe_ratio=ef.portfolio_performance()
        return portfolio_return, portfolio_volatility,sharpe_ratio
 
class PlotGraph:
    def plot_stock_data(dataFrame,title:str='Stock Value Over Time'):
       
        # Create a Plotly figure
        fig = go.Figure()

        # Add trace for Open prices
        fig.add_trace(go.Scatter(
            x=dataFrame['Date'],
            y=dataFrame['Open'],
            mode='lines',
            name='Open',
            line=dict(color='blue')
        ))

        # Add trace for High prices
        fig.add_trace(go.Scatter(
            x=dataFrame['Date'],
            y=dataFrame['High'],
            mode='lines',
            name='High',
            line=dict(color='green')
        ))

        # Add trace for Low prices
        fig.add_trace(go.Scatter(
            x=dataFrame['Date'],
            y=dataFrame['Low'],
            mode='lines',
            name='Low',
            line=dict(color='red')
        ))

        # Add trace for Close prices
        fig.add_trace(go.Scatter(
            x=dataFrame['Date'],
            y=dataFrame['Close'],
            mode='lines',
            name='Close',
            line=dict(color='orange')
        ))

        # Customize layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Price Type",
            template="simple_white",
            xaxis=dict(
                rangeslider=dict(visible=True),  # Add a range slider for interactive zooming
                type="date"
            ),
            height=600
        )
        # Show the plot
        fig.show()
        
class CommonAnalysis:
    def GroupData(dataFrame,gropingCol:str):
        df_grouped=dataFrame.groupby(gropingCol)
        return df_grouped
    def read_csv_file(filePath:str):
        dataframe=pd.read_csv(filePath)  # read csv file
        dataframe=dataframe.loc[:,~dataframe.columns.str.contains('^Unnamed')]
        return dataframe
    
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# class SentimentAnalysis:
#     # Preprocessing
#     stop_words = set(stopwords.words('english'))
#     lemmatizer = WordNetLemmatizer()
#     def get_sentiment(headline):
#         analysis = TextBlob(headline)
#         if analysis.sentiment.polarity > 0:
#             return 'Positive'
#         elif analysis.sentiment.polarity < 0:
#             return 'Negative'
#         else:
#             return 'Neutral'
#     def preprocess_text(headline):
#         tokens = word_tokenize(headline.lower())  # Tokenization
#         tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]  # Lemmatization
#         tokens = [word for word in tokens if word not in stop_words]  # Stopword removal
#         return tokens
    

        
    
