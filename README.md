

<!-- commands and  steps -->
### Development Environment Setup
#### Install Python
+ Verify if python is installed on your machine `python --version`
+ If python is not installed go to [Python.org](https://www.python.org/downloads/windows/) to download the appropriate installed for windows
#### Create and Activate a Python Virtual Environment
+ Create a python virtual environment
```
python -m venv .venv-week1
```
+ Activate Virtual Environment: change to project directory and run
```
.venv-week1\Scripts\activate.bat
```
#### Git and GitHub
+ A public `git` [repo](https://github.com/mebratuteshale/financial-data-analysis) created
+ Clone repo to local machine 
```
git clone https://github.com/mebratuteshale/financial-data-analysis.git
```
#### Installing libraries and packages required for Data Analysis
+ Download and install build of [TA-lib](ta_lib-0.5.1-cp312-cp312-win_amd64.whl) from github
```
    pip install C:\ta_lib-0.5.1-cp312-cp312-win_amd64.whl
``` 
+ Install pandas     
```
    pip install pandas           
```
Install yfinance 
``` 
    pip install yfinance
```
+ Install pypfopt
```
pip install PyPortfolioOpt
```
#### Descriptive Statistics:
+ Publications Count by day
![alt text](https://github.com/mebratuteshale/financial-data-analysis/blob/main/screenshots/NumOfPubOverTime.png)
+ Publication frequency by day of the week
![Analyze publication frequency by day of the week](https://github.com/mebratuteshale/financial-data-analysis/blob/main/screenshots/PubFreqByDayoftheWeek.png)
+ Publication trends by month and year
![Analyze publication trends by month and year](https://github.com/mebratuteshale/financial-data-analysis/blob/main/screenshots/PubTrendsbyMonthAndYr.png)

#### Text Analysis(Sentiment analysis & Topic Modeling)
+ A sentiment analysis on headlines was performed to gauge the sentiment as positive, negative or neutral associated with the news. As shown in the chart most of the news are **neutral** and below **2000** have a **negative** sentiment.
![Analyze publication trends by month and year](https://github.com/mebratuteshale/financial-data-analysis/blob/main/screenshots/PubTrendsbyMonthAndYr.png)
+ Next Topic modeling has been performed to the news headlines. Before performing the actual topic modeling **Text Preprocessing** activities suchas stop word removal is done. The analysis returned the mostly used words in the headlines of the news dataset.

#### Publisher Analysis
+ The news dataset was filtered and grouped by the publishers. A barchart is used to show the number of articles published by each publisher, since the publishers are large in number, the publisher name will be difficult to recognize from the chart.

### Task 2: Quantitative analysis using `yfinance` and `ta-lib`
+ Activities completed:
  * Install `yfinance` and `ta-lib`
  * Load and create dataframe from each historical stock dataset
  * Define functions to calculate `RSI`, `EMA`, `MACD` and `SMA`
  * A Time Series chart created to show the Stock price over time for each ticker.
  ![Analyze publication trends by month and year](https://github.com/mebratuteshale/financial-data-analysis/blob/main/screenshots/HistoricalDataTrace.png)

  * `RSI`, `EMA`, `MACD` and `SMA` are calculated for some tickers and a line graph plotted to show how the **RSI** vs **Closing** are moving overtime.

  ### Task 3: Correlation between news and stock movement
  + Tasks Completed: 
    * Date Alignment: normalizing timestamps of the historical data to **utc** for consistency
    * Set the Date column as the Index for each dataframe
    * Random Headline texts were selected from the news analyst dataframe.
    * Sentiment analysis of the headline texts is performed
    * Sample Correlations were calculated headline sentiment and stock Closing
    



