###
# get_data_2 is updated to use the alpha vantage API instead of yfinance as this is now depreciated and is no longer working correctly.
###
import csv
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
#supresses pandas warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import sqlite3
from tqdm import tqdm
import os

#Only gets last full days worth of data
def get_daily(symbol):
    ts = TimeSeries(key='BOH0A1X1EXCT3F1Y',output_format='pandas')
    # Get json object with the intraday data and another with  the call's metadata
    data, meta_data = ts.get_daily_adjusted(symbol = str(symbol), outputsize='compact')
    data = data.drop(columns=['5. adjusted close', '7. dividend amount', '8. split coefficient'])
    data.head()

    return data

#Gets the last months trading data minute by minuite
def get_historic_data(symbol):
    #Panadas Formatt not supported for intraday_extended
    ts = TimeSeries(key='BOH0A1X1EXCT3F1Y',output_format='csv')
    data = ts.get_intraday_extended(symbol = str(symbol),interval='1min', slice = 'year1month1')
    df = pd.DataFrame(list(data[0]))
    df.columns = df.iloc[0] #turn 1st row into headers
    df = df.drop(0)
    df.set_index('time',inplace = True)

    df.head()

    return df

def get_price(symbol):
    #Needs scrapping from somehwere no API SUPPORT
    print()

def get_overview(symbol):
    fd = FundamentalData(key='BOH0A1X1EXCT3F1Y')
    data = fd.get_company_overview(symbol)
    data = pd.Series(data[0])
    #The filter out useless information
    data = data[['MarketCapitalization','EBITDA','PERatio', 'PEGRatio','BookValue','DividendPerShare','DividendYield','EPS','RevenuePerShareTTM','ProfitMargin','OperatingMarginTTM','ReturnOnAssetsTTM','ReturnOnEquityTTM','RevenueTTM','GrossProfitTTM','QuarterlyEarningsGrowthYOY','QuarterlyRevenueGrowthYOY','AnalystTargetPrice' ,'EVToRevenue','52WeekHigh','52WeekLow','50DayMovingAverage','200DayMovingAverage','SharesOutstanding']]

    print(data)
    return data

def connect_to_db():
    # connect to Databse
    con = sqlite3.connect('Main.db')
    cursor = con.cursor()
    

    print('__ CONFIGURING TABLES __')

    #Create /connect to TRADE_DATA table which stores minute by minute trades over the last month +
    #Check to see if table already exists
    try:
        cursor.execute('''SELECT 1 FROM TRADE_DATA;''')
    #If not create DB
    except Exception as e:
        print('TRADE_DATA NOT FOUND CONFIGURING')
        cursor.execute('''CREATE TABLE TRADE_DATA
                    (Symbol TEXT,
                    Date TEXT,
                    Open REAL,
                    High REAL,
                    Low REAL,
                    Close REAL,
                    Volume REAL,
                    PRIMARY KEY (Symbol, Date))''')

    try:
        cursor.execute('''SELECT 1 FROM DAILY;''')
    except Exception as e:
        print('DAILY NOT FOUND CONFIGURING')
        cursor.execute('''CREATE TABLE DAILY
                    (Symbol TEXT,
                    Date TEXT,
                    Open REAL,
                    High REAL,
                    Low REAL,
                    Close REAL,
                    Volume REAL,
                    PRIMARY KEY (Symbol, Date))''')

    try:
        cursor.execute('''SELECT 1 FROM OVERVIEW;''')
    except Exception as e:
        print('OVERVIEW NOT FOUND CONFIGURING')
        cursor.execute('''CREATE TABLE OVERVIEW
                    (Symbol TEXT PRIMARY KEY,
                    MarketCapitalization INTEGER,
                    EBITDA INTEGER, 
                    PERatio REAL,
                    PEGRatio REAL,
                    BookValue REAL,
                    DividendPerShare REAL,
                    DividendYield REAL,
                    EPS REAL,
                    RevenuePerShareTTM REAL,
                    ProfitMargin REAL,
                    OperatingMarginTTM REAL,
                    ReturnOnAssetsTTM REAL,
                    ReturnOnEquityTTM REAL,
                    RevenueTTM REAL,
                    GrossProfitTTM REAL,
                    QuarterlyEarningsGrowthYOY REAL,
                    QuarterlyRevenueGrowthYOY REAL,
                    AnalystTargetPrice REAL,
                    EVToRevenue REAL,
                    FiftyTwoWeekHigh REAL,
                    FiftyTwoWeekLow REAL,
                    FiftyDayMovingAverage REAL,
                    TwoHunderedDayMovingAverage REAL,
                    SharesOutstanding REAL)''')

    return con,cursor

def Insert_Into_db(cursor,con,Values,SQL):
   
    try:
        cursor.execute(SQL, Values)
        print(Values)
        con.commit()
    except Exception as e:
        if str(e) == "UNIQUE constraint failed: TRADE_DATA.Symbol, TRADE_DATA.Date" or str(e) == "UNIQUE constraint failed: DAILY.Symbol, DAILY.Date" or str(e) == "UNIQUE constraint failed: OVERVIEW.Symbol":
            return
        else:
            print("ERROR when inserting Values: {}".format(e))
            return False

    return True


def main(symbol):
    
    

    con , cursor = connect_to_db()

    ### Update/get Company Overview
    print("Updating company ovierview for {}".format(symbol)) 
    data = get_overview(symbol)
    values = []
    values.append(symbol)
    values.extend(data.tolist())
    cursor.execute('DELETE FROM OVERVIEW WHERE Symbol == "{}";'.format(symbol))
    Insert_Into_db(cursor,con,values,'''Insert INTO OVERVIEW(Symbol,MarketCapitalization,EBITDA,PERatio,PEGRatio,BookValue,DividendPerShare,DividendYield,EPS,RevenuePerShareTTM,ProfitMargin,OperatingMarginTTM,ReturnOnAssetsTTM,ReturnOnEquityTTM,RevenueTTM,GrossProfitTTM,QuarterlyEarningsGrowthYOY,QuarterlyRevenueGrowthYOY,AnalystTargetPrice,EVToRevenue,FiftyTwoWeekHigh,FiftyTwoWeekLow,FiftyDayMovingAverage,TwoHunderedDayMovingAverage,SharesOutstanding) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ''')

    ### Update/get last months minuite by minuite trades
    print("inserting the following records for {} :".format(symbol))
    data = get_historic_data(symbol)
    for row in data.iterrows():
        values = []
        values.append(symbol)
        values.append(row[0])
        values.extend(row[1].tolist())
        Insert_Into_db(cursor,con,values,'''Insert INTO TRADE_DATA(Symbol,Date,Open,High,Low,Close,Volume) VALUES(?,?,?,?,?,?,?) ''')

    ### Update/Get day by day prices from the last 100 days
    daily = get_daily(symbol)
    for row in daily.iterrows():
        values = []
        values.append(symbol)
        values.append(row[0].strftime('%Y-%m-%d %X'))
        values.extend(row[1].tolist())
        Insert_Into_db(cursor,con,values,'''Insert INTO DAILY(Symbol,Date,Open,High,Low,Close,Volume) VALUES(?,?,?,?,?,?,?) ''')

    con.close()
    print("DATA UPDATED")
    
    return

if __name__ == '__main__':
    main("TSLA")