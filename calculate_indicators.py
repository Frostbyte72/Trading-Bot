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
def calc_atr(stock,interval):
    data = query('SELECT * FROM DAILY WHERE Symbol ="{}" ORDER BY Date ASC'.format(stock) )

    data['ATR'] = ta.ATR(data['High'],data['Low'],data['Close'],interval)

    return data['ATR']

def calc_ema(stock,interval):
    data = query('SELECT * FROM DAILY WHERE Symbol ="{}" ORDER BY Date ASC'.format(stock) )

    data['EMA'] = ta.EMA(data['Close'], interval)


    return data['EMA']

def price_sma(stock,interval):
    data = query('SELECT * FROM DAILY WHERE Symbol ="{}" ORDER BY Date ASC'.format(stock) )
    data['SMA'] = data['Close'].rolling(window=interval).mean()

    return data['SMA']

def volume_sma(stock,interval):
    data = query('SELECT * FROM DAILY WHERE Symbol ="{}" ORDER BY Date ASC'.format(stock) )
    data['SMA'] = data['Volume'].rolling(window=interval).mean()

    return data['SMA']

#Takes a stock,date and interval then claculates the RSI for that interval moving forward.
def calc_rsi(symbol,interval):
    # https://en.wikipedia.org/wiki/Relative_strength_index
    # This is cutlers RSI not Wilders RSI as cutlers dosen't change based on the starting point

    data = query('SELECT * FROM DAILY WHERE Symbol = "{}" ORDER BY DATE ASC'.format(symbol))
    RSI = pd.Series()
    RSI = ta.RSI(data['Close'],interval)

    return RSI

def calc_adx(symbol,interval):
    data = query('SELECT * FROM DAILY WHERE Symbol = "{}" ORDER BY DATE ASC'.format(symbol))

    ADX = pd.Series()
    ADX = ta.ADX(data['High'],data['Low'], data['Close'], timeperiod=interval)
    ADX.apply(lambda x: x/100) #sets ADX to be between 0-1 instead of 0-100
    return ADX

def calc_macd(symbol):
    data = query('SELECT * FROM DAILY WHERE Symbol = "{}" ORDER BY DATE ASC'.format(symbol))

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


def main(symbol):
    indicators = query('SELECT * FROM DAILY WHERE Symbol = "{}" ORDER BY DATE ASC'.format(symbol))
    indicators["CLOSE_SMA_52"] = price_sma(symbol,52)
    indicators["CLOSE_SMA_14"] = price_sma(symbol,14)
    indicators["CLOSE_SMA_7"] = price_sma(symbol,7)
    indicators["VOLUME_SMA_14"] = volume_sma(symbol,14)
    indicators["ATR_14"] = calc_atr(symbol,14)
    indicators["ATR_7"] = calc_atr(symbol,7)
    indicators['EMA_14'] = calc_ema(symbol,14)
    indicators['EMA_7'] = calc_ema(symbol,7)
    indicators['RSI_52'] = calc_rsi(symbol,52)
    indicators['RSI_14'] = calc_rsi(symbol,14)
    indicators['RSI_7'] = calc_rsi(symbol,7)
    indicators['ADX_14'] = calc_adx(symbol,14)
    indicators['ADX_7'] = calc_adx(symbol,7)

    print("---- INDICATORS ----")
    print(indicators)

    return indicators

if __name__ == "__main__":
    main("GOOGL")