import pandas as pd
import numpy as np
import sqlite3
from calculate_indicators import query as query
from calculate_indicators import main as m
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow import keras
from tqdm import tqdm
import matplotlib.pyplot as plt
from get_data_3 import main as get_data
import copy
import time

def organise_data(symbol):
    
    fundemental = query('SELECT * FROM FUNDEMENTAL WHERE Symbol = "{}" ORDER BY fiscalDateEnding DESC'.format(symbol))
    fundemental.drop(labels='Symbol',axis=1,inplace=True)
    #Technical analysis needs storing in db
    #technical = query("SELECT * FROM DAILY ORDER BY DATE")
    technical = m(symbol)
    #Reverse the order of the dataframe as it starts at the oldest first 
    technical = technical.iloc[::-1]
    #join fundamental data to each day in technical

    last_report_no = 0
    dataset = pd.DataFrame()
    #set intial date in datetime datatype
    length = len(technical)-1 #-1 to account for zero indexing
    report_date = datetime.strptime(fundemental['fiscalDateEnding'].iloc[0], '%Y-%m-%d')
    for i,row in technical.iterrows():
        row_date = datetime.strptime(row['Date'], '%Y-%m-%d %X')

        #create the Target Value for the row
        row['Target'] = row['Close'] - technical['Close'].iloc[length-i-1]

        # | #
        # V # if the day is before the report then switch to the previous report 
        if row_date <= report_date:
            last_report_no += 1
            #no more reports left break
            if last_report_no > len(fundemental)-1:
                break
            #update date
            report_date = datetime.strptime(fundemental['fiscalDateEnding'].iloc[last_report_no], '%Y-%m-%d')

        row = row.append(fundemental.iloc[last_report_no])
        dataset = dataset.append(row,ignore_index=True)
    
    
    #removes the frist row for the last known day of trading as we don't yet know tommorows close so it can have no taget value and thus useless training
    dataset.drop(index = 0,axis=0,inplace=True)


    return dataset

def split_data(dataset):
    #fraction of training data to be used for testing
    test_split = 0.1

    split_point = int(dataset.shape[0] - (dataset.shape[0] * test_split))
    
    test_data = dataset[split_point:-1]


    training_data = dataset[0:split_point]

    return test_data, training_data

# --- Prepares the data for the neural network ---
# Cleans data by removing null values and data that dosn't change
# also scales the data using the min max range to ensure that for any datapoint x in the data set is -1<=x<=1
def prepare_data(dataset):
    #Convert price data to change in price to increase sationairty of the data
    dataset['ChangeIn_Close'] = dataset['Close'].diff(periods=-1)
    dataset['ChangeIn_Open'] = dataset['Open'].diff(periods=-1)
    dataset['ChangeIn_High'] = dataset['High'].diff(periods=-1)
    dataset['ChangeIn_Low'] = dataset['Low'].diff(periods=-1)
    dataset['ChangeIn_Volume'] = dataset['Volume'].diff(periods=-1)

    #drop last record as the vlaue will be null
    dataset.drop(index=len(dataset),axis=0,inplace = True)
    dataset.drop(labels=['Close','Open','Low','High','Volume'],inplace= True, axis =1)

    for col in dataset.columns:
        dataset[col].fillna(method = 'backfill',inplace = True) #Replace unkown variables with the last known observation
        dataset[col].fillna(value = 0,inplace = True) #if no last observation replace with zero
        

        #Normalise the Data
        min = dataset[col].min()
        
        
        #if min is <0 add i add the absoloute value of the min to all records so you only have poistive values whilst keeping the scale.
        if min <0:
            dataset[col] = dataset[col].apply(lambda x: x + abs(min))
        
        max = dataset[col].max()
        min = dataset[col].min() #needs recalculating as it may have changed
        #To stop divide zero errors
        if min == max:
            if abs(min) > 0:
                dataset[col] = dataset[col].apply(lambda x: 0)
            
            continue
        
        
        dataset[col] = dataset[col].apply(lambda x: (x-abs(min))/(abs(max)-abs(min)))
        
    print(dataset.head(5))

    return dataset

#Data needs reshaing as RNN requires sequence data
#in RNN, GRU and LSTM, input is of shape t, where N is the number of samples, T is length of time sequence and D is the number of features.
#so input(shape = (D,)) for ANN and Input(shape = (T,D)
def shape_data(training_data,time_step,columns):
    shaped_data = []
    for index,_ in training_data.iterrows():
        #append next timestep records
        temp = training_data.iloc[index-1:index-1+time_step,:].to_numpy()
        shaped_data.append(temp)
        
    #remove the last time_step elements from the lsit as they won't be the correct shape.
    shaped_data = shaped_data[:-time_step]
    
    shaped_data = np.reshape(shaped_data,(len(shaped_data),time_step,columns))
    print("Reshaped Data")
    print(len(shaped_data))

    return shaped_data

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    #creates transformer blocks creating the encoder
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Decoder layer ~ not quite sure how this bit works yet
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units: #MLP = multilayer perceptron
        x = layers.Dense(dim, activation="tanh")(x)
        x = layers.Dropout(mlp_dropout)(x)
    
    outputs = layers.Dense(1, activation="linear")(x) #outputing the proabaility of belonging to a certain class which is why the loss bassicaly doesn't move becase the output is alwasy between 0-1
    
    return keras.Model(inputs, outputs)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    #This bit works it is learning which parts of the sequence to pay attention too
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    #This bit dosnt work its trying to classify the price into already seen values hence the convoloution layers
    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

    return x + res


#Arguments Decscirptions:
#-----------------------
# Symbol = String - Ticker of the stock/fund (List avalible at: https://www.alphavantage.co/documentation/)
# explain = Boolean - explain the features importance of the model
# plot = Boolean - Plot the Models prediction against actual values
# epochs = int - No. of epochs
# batch_size  = int - size of batch
# time_step = int - how many previous days tradying are provided to the LSTM model (time step of 7 will mean that the last rolling weeks data is used to predict the price)
# All other arguments are optional arguments for the keras model creation 
def main(tickers,update = False,plot= False,epochs = 150, batch_size =10,time_step = 7,loss_function = 'mse',optimiser = 'adam',learning_rate = 0.001,layer_size = 32, activation = 'tanh',dropout = 0.1):
    
    dataset = np.array(())
    count = 1
    target = pd.Series()
    for ticker in tickers['Symbol']:
        print('dataset Size: {}'.format(dataset.shape))
        print('Ticker Count: {}'.format(count))
        print('Ticker : {}'.format(ticker))
        if update:
            """if count/2 == int((count/2)):
                print('waiting for call limit to restart')
                time.sleep(60)"""
            print('Update is on')
            try:
                #update records
                print('Trying')
                get_data(ticker)
                ticker_data = organise_data(ticker)
            except Exception as e:
                e = str(e)
                if e[1:8] == 'None of':
                    print('EXCEDED API CALL LIMIT')
                    time.sleep(60)
                    get_data(ticker)
                    ticker_data = organise_data(ticker)
                else:
                    print('Exception')
                    raise Exception(e)
        else:
            ticker_data = organise_data(ticker)

        #Change target to daily difference in price
        ticker_data['Target'] = ticker_data['Close'].diff(periods=-1)
        temp = ticker_data['Target'].iloc[0:-(time_step+1)] #plus one to account for record that is dropped form the back beacause of the change in price etc.
        target = pd.concat([target,temp])

        #remove un-needed columns
        ticker_data.drop(labels=['Target','Symbol','Date','fiscalDateEnding'],axis = 1,inplace =True)

        #prepare and clean the data
        ticker_data = prepare_data(ticker_data)
        columns = ticker_data.shape[-1] #the size of the datframe's columns after redundent columns have been removed

        #Reshape Data
        x = shape_data(ticker_data,time_step,columns) 
        
        if count == 1:
            dataset = copy.deepcopy(x)
        else:
            dataset = np.concatenate((dataset,x),axis=0)

        
        #limit amunt of companies added to the dataset mainly used for testing
        if count == 1000:
            break

        count += 1
    

    model = build_model((time_step,columns),head_size =64,num_heads = 4,ff_dim = 4,num_transformer_blocks = 4,mlp_units =[64,64,64],mlp_dropout=0.4,dropout=0.2)

    model.compile(
    loss="mse",
    optimizer='adam',
    metrics=['mae','mse','mape'],
    )

    model.summary()

    model.fit(dataset, target, epochs = epochs, batch_size = batch_size)

    print('==================================')
    print('           Evaluation')
    print('----------------------------------')
    results = model.evaluate(dataset,target)
    predictions = model.predict(dataset)
    if plot:

        plt.plot(range(0,dataset.shape[0]),target, label = 'Actual Price',color = 'green' )
        plt.plot(range(0,dataset.shape[0]),predictions, label = 'Predicted Price',color = 'blue')
        plt.title('Stock GPT: Actual vs Predicted Change in Price')
        plt.ylabel('Daily Change in Price $ (USD)')
        plt.legend()
        plt.show()

    model.save('Stock_GPT_NASDAQ100_ts128')

    return model,results

if __name__ == '__main__':
    #tickers = pd.read_csv('SNP500.csv')
    tickers = pd.read_csv('NASDAQ_100.csv')
    print(tickers.head(5))
    main(tickers,update = False ,plot = True,epochs = 150,time_step = 128,batch_size = 10,activation = 'tanh')
