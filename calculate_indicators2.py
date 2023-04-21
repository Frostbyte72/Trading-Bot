import sqlite3
import pandas as pd
#supresses pandas warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import datetime
from get_data_2 import Insert_Into_db
import talib as ta

#Explination of indicators
#https://www.ig.com/uk/trading-strategies/10-trading-indicators-every-trader-should-know-190604

#Calculates the 12 day exponential moving average
#trend over 12 days shows more recent price trends
#EMA = Closing price x multiplier + EMA (previous day) x (1-multiplier)
#Multiplier = (2/((Number of ovbservations) +1))
def calc_atr(interval,data):

    data['ATR'] = ta.ATR(data['High'],data['Low'],data['Close'],interval)

    return data['ATR']

def calc_ema(interval,data):

    data['EMA'] = ta.EMA(data['Close'], interval)


    return data['EMA']

def price_sma(interval,data):

    data['SMA'] = data['Close'].rolling(window=interval).mean()

    return data['SMA']

def volume_sma(interval,data):
    data['SMA'] = data['Volume'].rolling(window=interval).mean()

    return data['SMA']

#Takes a stock,date and interval then claculates the RSI for that interval moving forward.
def calc_rsi(interval,data):
    # https://en.wikipedia.org/wiki/Relative_strength_index
    # This is cutlers RSI not Wilders RSI as cutlers dosen't change based on the starting point

    RSI = pd.Series()
    RSI = ta.RSI(data['Close'],interval)

    return RSI

def calc_adx(interval,data):

    ADX = pd.Series()
    ADX = ta.ADX(data['High'],data['Low'], data['Close'], timeperiod=interval)
    ADX.apply(lambda x: x/100) #sets ADX to be between 0-1 instead of 0-100
    return ADX

def calc_macd(data):

    MACD = pd.Series()
    MACD = ta.MACD(data['Close'])

    return MACD

#query function for sql database
#Reuturns rows as pd dataframe
def query(query):
    con = sqlite3.connect('Main.db')
    #cursor = con.cursor()

    try:
        data = pd.read_sql_query(query, con)
        con.close()
        return data
    except Exception as e:
        print("ERROR WHILE EXECUTING SQL: {}".format(e))

    con.close()
    return


def main(dataset):
    print(dataset.head(5))
    dataset["CLOSE_SMA_52"] = price_sma(52,dataset)
    dataset["CLOSE_SMA_14"] = price_sma(14,dataset)
    dataset["CLOSE_SMA_7"] = price_sma(7,dataset)
    dataset["VOLUME_SMA_14"] = volume_sma(14,dataset)
    dataset["ATR_14"] = calc_atr(14,dataset)
    dataset["ATR_7"] = calc_atr(7,dataset)
    dataset['EMA_14'] = calc_ema(14,dataset)
    dataset['EMA_7'] = calc_ema(7,dataset)
    dataset['RSI_52'] = calc_rsi(52,dataset)
    dataset['RSI_14'] = calc_rsi(14,dataset)
    dataset['RSI_7'] = calc_rsi(7,dataset)
    dataset['ADX_14'] = calc_adx(14,dataset)
    dataset['ADX_7'] = calc_adx(7,dataset)
    dataset.drop(labels=['SMA','EMA','ATR'],axis =1, inplace = True)

    print("---- INDICATORS ----")
    print(dataset)

    return dataset

if __name__ == "__main__":
    stock = 'GOOGL'
    data = query('SELECT * FROM DAILY WHERE Symbol ="{}" ORDER BY Date ASC'.format(stock) )
    main(data)