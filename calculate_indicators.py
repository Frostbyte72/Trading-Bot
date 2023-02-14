import sqlite3
import pandas as pd
import datetime

#Explination of indicators
#https://www.ig.com/uk/trading-strategies/10-trading-indicators-every-trader-should-know-190604

#Calculates the Moving range (200 day)
#price trend over a long period of time
# avgMR = sum(x1 - x2, ... xk-1 - xk) / k-1
def calc_mr(stock):
    data = query('SELECT * FROM TRADE_DATA WHERE Symbol ="{}" LIMIT 200'.format(stock) )
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

def calc_rsi(symbol,date,interval):
    #https://www.fmlabs.com/reference/default.htm?url=RSI.htm
    current_date = datetime.datetime.strptime(date, '%Y-%m-%d %X')
    print(current_date)
    for i in range (0,interval):
        data = query('SELECT * FROM DAILY WHERE Symbol = "{}" AND Date = "{}"'.format(symbol,date))
        if data == None:
            print("There is a gap in the data consider changing interval (possible holiday)")
        current_date = current_date + datetime.timedelta(days=1)
        print(data)
         
    return 

#query function for sql database
#Reuturns rows as pd dataframe
def query(query):
    con = sqlite3.connect('Main.db')
    #cursor = con.cursor()

    try:
        data = pd.read_sql_query(query, con)
        return data
    except Exception as e:
        print(e)

    return

def main():
    symbol = "GOOGL"
    date = "2023-01-20 00:00:00" 
    print(calc_mr(symbol))
    print(calc_ema(symbol))
    print(calc_rsi(symbol,date,14))
  
    


if __name__ == "__main__":
    main()