import yfinance as yf
import talib as ta
import pandas as pd
import numpy as np
import plotly.express as px
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

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

class GeneralConfiguration:
    def read_csv_file(filePath:str):
        data=pd.read_csv(filePath)  # read csv file
        data=data.loc[:,~data.columns.str.contains('^Unnamed')]
        return data
    
class PlotGraph:
    def plot_stock_data(dataFrame,dateColumn:str='date',stockValueCol:str='stock_value',title:str='Stock Value Over Time'):
        # Plots stock value over time from csv file
        dataFrame[dateColumn]=pd.to_datetime(dataFrame[dateColumn],errors='coerce',format='%Y-%m-%d %H:%M:%S')
        dataFrame.dropna(subset=[dateColumn],inplace=True)
        plt.figure(figsize=(10,6))
        plt.plot(dataFrame[dateColumn],dataFrame[stockValueCol],label=stockValueCol,color='b')
        plt.xlabel('Date')
        plt.ylabel('Stock Value')
        plt.title(title)

        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    
