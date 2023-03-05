import NN_Model_2
import get_data_2

symbol = input("Ticker of company to analyse: ")

get_data_2.main(symbol)
NN_Model_2.main(symbol)


print("Model created")