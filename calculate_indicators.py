import sqlite3
import pandas as pd
#supresses pandas warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import datetime
from get_data_2 import Insert_Into_db
import talib

#Explination of indicators
#https://www.ig.com/uk/trading-strategies/10-trading-indicators-every-trader-should-know-190604

#Calculates the Moving range
#price trend over a long period of time
# avgMR = sum(x1 - x2, ... xk-1 - xk) / k-1
def calc_mr(stock,limit):
    #needs sorting by date so it gets the last 200
    data = query('SELECT * FROM DAILY WHERE Symbol ="{}" ORDER BY Date DESC LIMIT {}'.format(stock,limit) )
    sum = 0
    for index in range(0,data.shape[0]-1):
        sum = sum + (float(data['Close'].iloc[index]) - float(data['Close'].iloc[index+1]))

    mr = sum/(data.shape[0]-1)
    return mr

#Calculates the 12 day exponential moving average
#trend over 12 days shows more recent price trends
#EMA = Closing price x multiplier + EMA (previous day) x (1-multiplier)
#Multiplier = (2/((Number of ovbservations) +1))
def calc_ema(stock):
    data = query('SELECT * FROM TRADE_DATA WHERE Symbol ="{}" LIMIT 12'.format(stock) )
    observations = 0
    prev_ema = 0
    for i in range(0,data.shape[0]):
        multiplier = (2/(observations+1))
        EMA = float(data['Close'].iloc[i]) * multiplier + prev_ema * (1-multiplier)
        observations += 1

    return EMA

def price_sma(stock,interval):
    data = query('SELECT * FROM DAILY WHERE Symbol ="{}" ORDER BY Date ASC'.format(stock) )
    data['SMA'] = data['Close'].rolling(window=interval).mean()

    return data

def volume_sma(stock,interval):
    data = query('SELECT * FROM DAILY WHERE Symbol ="{}" ORDER BY Date ASC'.format(stock) )
    data['SMA'] = data['Volume'].rolling(window=interval).mean()

    return data

#Takes a stock,date and interval then claculates the RSI for that interval moving forward.
#can get stuck in a infinite loop if it cant find data points within the intervale due to while loops.
# as such Interval for RSI should be retrive as a count of records in the databse to avoid infinite loop
def calc_rsi(symbol,date,interval):
    # https://en.wikipedia.org/wiki/Relative_strength_index
    # This is cutlers RSI not Wilders RSI as cutlers dosen't change based on the starting point
    current_date = datetime.datetime.strptime(date, '%Y-%m-%d %X')
    u = []
    d = []
    for i in range (0,interval-1):
        point = pd.Series()
        next_point = pd.Series()
        # next point is current price 
        
        #needed to skip days where the market is closed
        while point.empty:
            point = query('SELECT * FROM DAILY WHERE Symbol = "{}" AND Date = "{}"'.format(symbol,current_date))
            current_date = current_date + datetime.timedelta(days=1)
        while next_point.empty:
            next_point = query('SELECT * FROM DAILY WHERE Symbol = "{}" AND Date = "{}"'.format(symbol,current_date))
            current_date = current_date + datetime.timedelta(days=1)

        if float(point["Close"]) < float(next_point["Close"]):#next point's close
            #price has gone up
            u.append(next_point["Close"] - point["Close"])
            d.append(0)
        elif float(point["Close"]) > float(next_point["Close"]):
            #Price has gone down
            u.append(0)
            d.append(point["Close"] - next_point["Close"])
        else:
            #Price is the same
            u.append(0)
            d.append(0)

    #Calcuate SMA of u and d

    utotal = 0
    dtotal = 0
    for i in range(0,(interval -1)):
        utotal += u[i]
        dtotal += d[i]

    SMAU = utotal/len(u)
    SMAD = dtotal/len(d)

    RSI = 100 * SMAU/(SMAU+SMAD)

    print(RSI)
    return float(RSI)


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


def main():
    symbol = "GOOGL"
    date = "2023-01-01 00:00:00"
    print(price_sma(symbol,14)) 
    print(volume_sma(symbol,14))
    print(calc_mr(symbol,14))
    print(calc_ema(symbol))
    #Interval for RSI should be retrive as a count of records in the databse to avoid infinite loop
    print(calc_rsi(symbol,date,14))


if __name__ == "__main__":
    main()