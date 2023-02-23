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
import requests

#Only gets last full days worth of data
def get_daily(symbol):
    ts = TimeSeries(key='BOH0A1X1EXCT3F1Y',output_format='pandas')
    # Get json object with the intraday data and another with  the call's metadata
    data, meta_data = ts.get_daily_adjusted(symbol = str(symbol), outputsize='full')
    
    #Scale other values to account for stock splits
    adjusted_data = pd.DataFrame()
    for i, row in data.iterrows():
        row['Date'] = i.strftime('%Y-%m-%d %X')
        split_coefficient = (float(row['5. adjusted close']) / float(row['4. close']))
        row['1. open'] = row['1. open'] * split_coefficient
        row['2. high'] = row['2. high'] * split_coefficient
        row['3. low'] = row['3. low'] * split_coefficient    
        row['6. volume'] = row['6. volume'] * split_coefficient
        adjusted_data = adjusted_data.append(row,ignore_index=True)
    
    #set close as the adjusted close to account for stock splits
    adjusted_data['4. close'] = adjusted_data['5. adjusted close']

    #drop unessecary columns
    adjusted_data = adjusted_data.drop(columns=['5. adjusted close', '7. dividend amount', '8. split coefficient'])
    return adjusted_data

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

####### Needs changing to add in historical overviews for fundemnetal analyasis for the day to day price prediction. ######
def get_overview(symbol):
    fd = FundamentalData(key='BOH0A1X1EXCT3F1Y')
    data = fd.get_company_overview(symbol)
    data = pd.Series(data[0])
    #The filter out useless information
    data = data[['MarketCapitalization','EBITDA','PERatio', 'PEGRatio','BookValue','DividendPerShare','DividendYield','EPS','RevenuePerShareTTM','ProfitMargin','OperatingMarginTTM','ReturnOnAssetsTTM','ReturnOnEquityTTM','RevenueTTM','GrossProfitTTM','QuarterlyEarningsGrowthYOY','QuarterlyRevenueGrowthYOY','AnalystTargetPrice' ,'EVToRevenue','52WeekHigh','52WeekLow','50DayMovingAverage','200DayMovingAverage','SharesOutstanding']]

    print(data)
    return data

def get_income_statments(symbol):
    url = 'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={}&apikey=BOH0A1X1EXCT3F1Y'.format(symbol)
    r = requests.get(url)
    raw_data = r.json()
    data = pd.DataFrame(raw_data.get("quarterlyReports"))
    data = data[['fiscalDateEnding','grossProfit','totalRevenue','operatingIncome','netIncome','ebitda']]

    print(data)
    return data

def get_balance_sheets(symbol):
    url = 'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={}&apikey=BOH0A1X1EXCT3F1Y'.format(symbol)
    r = requests.get(url)
    raw_data = r.json()

    data = pd.DataFrame(raw_data.get("quarterlyReports"))
    data = data[['fiscalDateEnding','totalAssets','totalCurrentAssets','investments','totalLiabilities','totalShareholderEquity','treasuryStock','retainedEarnings','commonStock','commonStockSharesOutstanding']]


    print(data)

    return data

def get_cash_flow(symbol):
    url = 'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={}&apikey=BOH0A1X1EXCT3F1Y'.format(symbol)
    r = requests.get(url)
    raw_data = r.json()
    
    data = pd.DataFrame(raw_data.get("quarterlyReports"))
    
    data = data[['fiscalDateEnding','operatingCashflow','profitLoss','paymentsForRepurchaseOfCommonStock','paymentsForRepurchaseOfPreferredStock','dividendPayout','proceedsFromIssuanceOfCommonStock','proceedsFromIssuanceOfPreferredStock','cashflowFromInvestment','changeInCashAndCashEquivalents']]

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
    except:
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
    except:
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
    except:
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

    try:
        cursor.execute('''SELECT 1 FROM FUNDEMENTAL;''')
    except:
        print('OVERVIEW NOT FOUND CONFIGURING')
        cursor.execute('''CREATE TABLE FUNDEMENTAL
                    (Symbol TEXT,
                    fiscalDateEnding TEXT,
                    grossProfit REAL,
                    totalRevenue REAL,
                    operatingIncome REAL,
                    netIncome REAL,
                    ebitda REAL,
                    totalAssets REAL,
                    totalCurrentAssets REAL,
                    investments REAL,
                    totalLiabilities REAL,
                    totalShareholderEquity REAL,
                    treasuryStock REAL,
                    retainedEarnings REAL,
                    commonStock INTEGER,
                    commonStockSharesOutstanding INTEGER,
                    operatingCashflow REAL,
                    profitLoss REAL,
                    paymentsForRepurchaseOfCommonStock REAL,
                    paymentsForRepurchaseOfPreferredStock REAL,
                    dividendPayout REAL,
                    proceedsFromIssuanceOfCommonStock REAL,
                    proceedsFromIssuanceOfPreferredStock REAL, 
                    cashflowFromInvestment REAL,
                    changeInCashAndCashEquivalents REAL ,
                    ROE REAL,
                    profitMargin REAL,
                    ROI REAL,
                    DPS REAL,
                    PRIMARY KEY (Symbol, fiscalDateEnding))''')


    return con,cursor

def Insert_Into_db(cursor,con,Values,SQL):
   
    try:
        cursor.execute(SQL, Values)
        print(Values)
        con.commit()
    except Exception as e:
        if str(e) == "UNIQUE constraint failed: TRADE_DATA.Symbol, TRADE_DATA.Date" or str(e) == "UNIQUE constraint failed: DAILY.Symbol, DAILY.Date" or str(e) == "UNIQUE constraint failed: OVERVIEW.Symbol" or str(e) == "UNIQUE constraint failed: FUNDEMENTAL.Symbol, FUNDEMENTAL.fiscalDateEnding":
            return True
        else:
            print("ERROR when inserting Values: {}".format(e))
            return False

    return True


def main(symbol):

    con , cursor = connect_to_db()

    #Collecting Fundemental anyalsis data
    income = get_income_statments(symbol)
    balance = get_balance_sheets(symbol)
    cash = get_cash_flow(symbol)
    #join the fundmental data together
    temp = pd.merge(income,balance, how="left", on="fiscalDateEnding")
    fundamentals = pd.merge(temp,cash, how="left", on="fiscalDateEnding")
    print(fundamentals)
    
    #0 = return on equity (ROE) , 1 = Profit Margin ,2 = return on investment (ROI), 3 = dividends per share (DPS)
    calculated = [[],[],[],[]]
    for i in range(0,fundamentals.shape[0]):
        ROE = float(fundamentals['netIncome'].iloc[i])/( (float(fundamentals['totalAssets'].iloc[i])) - float(fundamentals['totalLiabilities'].iloc[i]) )
        calculated[0].append(ROE)
        
        profit_margin = float(fundamentals['grossProfit'].iloc[i] ) / float(fundamentals['totalRevenue'].iloc[i])
        calculated[1].append(profit_margin)

        ROI = float(fundamentals['cashflowFromInvestment'].iloc[i]) / float(fundamentals['investments'].iloc[i])
        calculated[2].append(ROI)

        if fundamentals['dividendPayout'].iloc[i] == 'None':
            DPS = 0
        else:
            DPS = float(fundamentals['dividendPayout'].iloc[i]) / (float(fundamentals['commonStock'].iloc[i]))
        calculated[3].append(DPS)

    fundamentals['ROE'] = calculated[0]
    fundamentals['profitMargin'] = calculated[1]
    fundamentals['ROI'] = calculated[2]
    fundamentals['DPS'] = calculated[3]

    for row in fundamentals.iterrows():
        values = []
        values.append(symbol)
        for i in range(0,row[1].size):
            if row[1].iloc[i] == 'None':
                row[1].iloc[i] =  None
            else:
                continue
        values.extend(row[1].tolist())
        
        Insert_Into_db(cursor,con,values,'''Insert INTO FUNDEMENTAL(Symbol,fiscalDateEnding, grossProfit, totalRevenue, operatingIncome, netIncome, ebitda, totalAssets, totalCurrentAssets, investments, totalLiabilities, totalShareholderEquity, treasuryStock, retainedEarnings, commonStock, commonStockSharesOutstanding, operatingCashflow, profitLoss, paymentsForRepurchaseOfCommonStock, paymentsForRepurchaseOfPreferredStock, dividendPayout, proceedsFromIssuanceOfCommonStock, proceedsFromIssuanceOfPreferredStock, cashflowFromInvestment, changeInCashAndCashEquivalents, ROE, profitMargin, ROI, DPS) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ''')


    ### Update/get Company Overview
    """
    print("Updating company ovierview for {}".format(symbol)) 
    data = get_overview(symbol)
    values = []
    values.append(symbol)
    values.extend(data.tolist())
    cursor.execute('DELETE FROM OVERVIEW WHERE Symbol == "{}";'.format(symbol))
    Insert_Into_db(cursor,con,values,'''Insert INTO OVERVIEW(Symbol,MarketCapitalization,EBITDA,PERatio,PEGRatio,BookValue,DividendPerShare,DividendYield,EPS,RevenuePerShareTTM,ProfitMargin,OperatingMarginTTM,ReturnOnAssetsTTM,ReturnOnEquityTTM,RevenueTTM,GrossProfitTTM,QuarterlyEarningsGrowthYOY,QuarterlyRevenueGrowthYOY,AnalystTargetPrice,EVToRevenue,FiftyTwoWeekHigh,FiftyTwoWeekLow,FiftyDayMovingAverage,TwoHunderedDayMovingAverage,SharesOutstanding) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ''')"""

    #Collecting data for Techincal anyalsis
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
    for _,row in daily.iterrows():
        values = []
        values.append(symbol)
        values.extend(row.tolist())
        Insert_Into_db(cursor,con,values,'''Insert INTO DAILY(Symbol,Open,High,Low,Close,Volume,Date) VALUES(?,?,?,?,?,?,?) ''')

    con.close()
    print("DATA UPDATED")
    
    return

if __name__ == '__main__':
    main("GOOGL")