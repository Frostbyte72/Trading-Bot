import NN_Model
import get_data_2

symbol = input("Ticker of company to analyse: ")

get_data_2.main(symbol)
NN_Model.main(symbol)


print("Model created")