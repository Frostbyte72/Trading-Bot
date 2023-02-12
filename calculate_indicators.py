import sqlite3
import pandas as pd

#Explination of indicators
#https://www.ig.com/uk/trading-strategies/10-trading-indicators-every-trader-should-know-190604

#Calculates the Moving range (200 day)
# avgMR = sum(x1 - x2, ... xk-1 - xk) / k-1
def calc_mr(stock):
    data = query('SELECT * FROM PRICES WHERE NAME ="{}" LIMIT 200'.format(stock) )
    sum = 0
    for index in range(0,data.shape[0]-1):
        print(index)
        sum = sum + (float(data['Close'].iloc[index]) - float(data['Close'].iloc[index+1]))

    mr = sum/(data.shape[0]-1)
    return mr


#query function for sql database
#Reuturns rows as pd dataframe
def query(query):
    con = sqlite3.connect('Market_data.db')
    cursor = con.cursor()

    try:
        data = pd.read_sql_query(query, con)
    except Exception as e:
        print(e)

    return data

def main():
    print(calc_mr("AAAU"))


if __name__ == "__main__":
    main()