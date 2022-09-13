import yfinance as yf

msft = yf.Ticker("MSFT")

print(msft.cashflow)
print(msft.earnings)