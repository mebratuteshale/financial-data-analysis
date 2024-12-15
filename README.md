

<!-- commands and  steps -->
## Development Environment Setup
### Install Python
+ Verify if python is installed on your machine `python --version`
+ If python is not installed go to [Python.org](https://www.python.org/downloads/windows/) to download the appropriate installed for windows
### Create and Activate Python a Virtual Environment
+ Create a python virtual environment
```
python -m venv .venv-week1
```
+ Activate Virtual Environment: change to project directory and run
```
.venv-week1\Scripts\activate.bat
```
### Installing libraries and packages required for Data Analysis
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
## Descriptive Statistics:
+ Publications Count by day
![alt text](https://github.com/mebratuteshale/financial-data-analysis/blob/main/screenshots/NumOfPubOverTime.png)
+ Publication frequency by day of the week
![Analyze publication frequency by day of the week](https://github.com/mebratuteshale/financial-data-analysis/blob/main/screenshots/PubFreqByDayoftheWeek.png)
+ Publication trends by month and year
![Analyze publication trends by month and year](https://github.com/mebratuteshale/financial-data-analysis/blob/main/screenshots/PubTrendsbyMonthAndYr.png)
