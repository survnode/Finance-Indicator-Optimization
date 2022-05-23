import alpaca_trade_api as api
import backtrader as bt
import pandas as pd
import pytz
from credentials import alpaca_paper #store credentials separately
from tuneta.tune_ta import TuneTA
from pandas_ta import percent_return
from sklearn.model_selection import train_test_split

# Initiate API parameters
ALPACA_KEY_ID = alpaca_paper['api_key']
ALPACA_SECRET_KEY = alpaca_paper['api_secret']
ALPACA_PAPER = True
alpaca = api.REST(ALPACA_KEY_ID, ALPACA_SECRET_KEY)


# Set data parameters
TIMEZONE = pytz.timezone('America/New_York') #Specify your prefered timezone
timeframe = '1Day' # Bar timeframe  (Alternative timeframes xxMinute, xxHour, xxWeek, xxMonth etc)
start = pd.Timestamp('2021-01-01 00:00', tz=TIMEZONE).isoformat() #Start date for bar data
end = pd.Timestamp('2022-04-05 00:00', tz=TIMEZONE).isoformat() #End date for bar data

symbol = ['SPY']  #Target ticker. Option to include additional tickers e.g. ['SPY', "AAPL", "MSFT"] etc

# Retrieve target ticker symbol(s) within specified timeframe & parameters
X = alpaca.get_bars(symbol, timeframe, start, end).df
#X.to_csv('historical_prices.csv', sep=',') #Option to download the data to a .csv file or any format of your choice
X.head()


#Preprocess the data

#The dataframe does not have an index. The 'timestamp' has been 
# incorrectly assiged as the index so we need to first correct this.
#X = X.reset_index()
#Drop the last 3 columns
X.drop(columns=X.columns[-3:], axis=1, inplace=True)

# Split the timestamp column to date and time
#X['date'] = pd.to_datetime(X['timestamp']).dt.date
#X['time'] = pd.to_datetime(X['timestamp']).dt.time

#Reorder the dataframe for convention and drop the timestamp column
#X = X[['date', 'time', 'open', 'high', 'low', 'close', 'volume']]
X.head()

# Split the data into training and testing set and offset the close by -1
# so that the previous bar features predict the current close
y = percent_return(X.close, offset=-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, shuffle=False)

# Initialize with 6 cores and show trial results
tt = TuneTA(n_jobs=6, verbose=True)

# Optimize indicators
tt.fit(X_train, y_train, indicators=['all'], ranges=[(2, 30), (31, 180)], trials=500, early_stop=100,)

# Show time duration in seconds per indicator
tt.fit_times()

# Show correlation of indicators to target
tt.report(target_corr=True, features_corr=True)

# Select features with at most x correlation between each other
tt.prune(max_inter_correlation=.7)

# Show correlation of indicators to target and among themselves
tt.report(target_corr=True, features_corr=True)

# Add indicators to X_train
features = tt.transform(X_train)
X_train = pd.concat([X_train, features], axis=1)

# Add same indicators to X_test
features = tt.transform(X_test)
X_test = pd.concat([X_test, features], axis=1)
