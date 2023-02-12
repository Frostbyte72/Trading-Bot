import csv
import requests
import yfinance as yf
#supresses pandas warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import sqlite3
from tqdm import tqdm
import os
import keyboard

def get_price_data(symbol):
    stock = yf.Ticker(symbol)
   # get price History
    price_data = pd.DataFrame(stock.history(period="1y"))
    price_data.drop(['Dividends','Stock Splits'],1,inplace=True)
    return price_data


def get_stock_info(symbol):
    stock = yf.Ticker(symbol)
    raw_info = stock.info
    info = pd.Series(raw_info) 
    info.drop(info.index.difference(['symbol','longName','category','longBusinessSummary']), inplace=True)   
    
    #rearange data into correct format
    info = info.reindex(index = ['symbol','longName','category','longBusinessSummary'])

    return info

def get_price(symbol):
    stock = yf.Ticker(symbol)
    print(stock.info)
    market_price = stock.info['regularMarketPrice']

    return market_price

def Insert_Into_db(cursor,con,Values,SQL):
   
    try:
        cursor.execute(SQL, Values)
        con.commit()
    except Exception as e:
        if str(e) == "UNIQUE constraint failed: PRICES.Name, PRICES.Date":
            return
        else:
            print("ERROR when inserting Values: {}".format(e))
            return False

    return True


def get_symbols():
    df = pd.read_csv("symbols.csv")
    #print(df.head())
    symbols = df.iloc[:,0]

    return symbols


def connect_to_db():
    # connect to Databse
    con = sqlite3.connect('Market_data.db')
    cursor = con.cursor()

    print('__ CONFIGURING TABLES __')

    #Check to see if table already exists
    try:
        cursor.execute('''SELECT 1 FROM STOCKS;''')
    except Exception as e:
        print('Stocks NOT FOUND CONFIGURING')
        cursor.execute('''CREATE TABLE STOCKS
                    ( Name Text,
                    Symbol TEXT PRIMARY KEY,
                    Catagory TEXT,
                    Summary TEXT,
                    Sentiment REAL,
                    Weighted_Sentiment REAL)''')
    
    try:
        cursor.execute('''SELECT 1 FROM PRICES;''')
    except Exception as e:
        print('Prices NOT FOUND CONFIGURING')
        cursor.execute('''CREATE TABLE PRICES
                    (Name TEXT,
                    Date TEXT,
                    Open REAL,
                    High REAL,
                    Low REAL,
                    Close REAL,
                    Volume INTEGER,
                    PRIMARY KEY (Name, Date));''')

    try:
        cursor.execute('''SELECT 1 FROM DAY_PRICE''')
    except Exception as e:
        print('DAY_PRICE NOT FOUND CONFIGURING')
        cursor.execute('''CREATE TABLE DAY_PRICE
                    (Name TEXT,
                    Date TEXT,
                    price REAL,
                    PRIMARY KEY (Name, Date));''')

    print("DB SUCESSFULLY CONFIGURED")

    return con,cursor

def clear():
    os.system('cls' if os.name=='nt' else 'clear')
    return
    
def main():
    i = input("Initialise company data Setup? (y/n): ")
    symbols = get_symbols()

    con,cursor = connect_to_db()

    #Redownload all prices history from stocks in the CSV  
    if i == "y":
        for index, name in symbols.items():
            print("Getting Company Data for", name)
            info = get_stock_info(str(name))
            Values = info.tolist()
            Values.extend([None,None])
            if Values[1] == None:
                print("Record has no symbol cannot be inserted")
                break
            Insert_Into_db(cursor,con,Values,'''INSERT INTO STOCKS(Symbol,Name,Catagory,Summary,Sentiment,Weighted_Sentiment) VALUES(?,?,?,?,?,?) ''')
    
    i = input("Update Price Data ? (y/n): ")
    if i == "y":
        #Update all price data of existing stocks in the table
        print("---Updating Price Data---")
        #change to get name from sql database so that with mismatch in records price history still works
        cursor.execute("SELECT Symbol FROM Stocks")
        symbols = cursor.fetchall()
        
        for symbol in tqdm(symbols, desc = 'Update Progress'):
            #print("Getting Price history for", symbol[0])
            data = get_price_data(symbol[0])
            for row in data.iterrows():
                Values = []
                Values.append(symbol[0])
                Values.append(row[0].strftime('%Y-%m-%d %X'))
                Values.extend(row[1].tolist())
                Insert_Into_db(cursor,con,Values,'''Insert INTO Prices(Name,Date ,Open,High,Low,Close,Volume) VALUES(?,?,?,?,?,?,?) ''')
        print("---Avalible Price History Collected---")

    print("Current Price")
    price = get_price("AAAU")
    print(price)


if __name__ == '__main__':
    get_stock_info('MSFT')
    main()