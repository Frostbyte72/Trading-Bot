import pandas as pd
import numpy as np
import sqlite3
from calculate_indicators2 import query as query
from calculate_indicators2 import main as m
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


    return dataset

def add_fundemnetal_data(technical):
    symbol = technical['Symbol'].iloc[0]
    fundemental = query('SELECT * FROM FUNDEMENTAL WHERE Symbol = "{}" ORDER BY fiscalDateEnding DESC'.format(symbol))
    fundemental.drop(labels='Symbol',axis=1,inplace=True)

    last_report_no = 0
    dataset = pd.DataFrame()
    #set intial date in datetime datatype
    length = len(technical)-1 #-1 to account for zero indexing
    report_date = datetime.strptime(fundemental['fiscalDateEnding'].iloc[0], '%Y-%m-%d')
    for i,row in technical.iterrows():
        row_date = datetime.strptime(row['Date'], '%Y-%m-%d %X')

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

         ########################
    # Z score outlier removal
    
    mean = dataset['ChangeIn_Close'].iloc[-1200:-1].mean()
    std = dataset['ChangeIn_Close'].iloc[-1200:-1].std()
    threshold = 2
    dataset['Zscore'] = dataset['ChangeIn_Close'].apply(lambda x : (x-mean)/std) 
    print(dataset['Zscore'])
    print('3X Standard Deviation {}'.format(std*threshold))
    dataset = dataset[ (dataset['Zscore'] < threshold) & (dataset['Zscore'] > -threshold )]
    print(dataset['Zscore'].max())
    print(dataset['Zscore'].min())
    dataset.drop(labels = 'Zscore',axis = 1, inplace = True)
    #reset index as records are removed in the middle
    dataset.reset_index(inplace = True)
    


    #add techical indicators
    dataset = m(dataset)
    #Reverse the order of the dataframe as it starts at the oldest first 
    dataset = dataset.iloc[::-1]

    # Create the target in the dataset
    dataset['target'] = dataset['ChangeIn_Close'].shift(periods = -1)
    
    #drop last record as the value will be null due to the change in close
    dataset.drop(index=dataset.index[-1],axis=0,inplace = True)
    #removes the frist row for the last known day of trading as we don't yet know tommorows close so it can have no taget value and thus useless training
    dataset.drop(index =dataset.index[0],axis=0,inplace=True)


    #add Fundemental Data
    dataset = add_fundemnetal_data(dataset)
    
    #set target to its own variable so it can be excluded from the training data
    target = dataset['target']
    print(dataset.head(5))
    print(dataset.tail(5))

    #index
    dataset.drop(labels=['target','Close','Open','Low','High','Volume','Symbol','Date','fiscalDateEnding'],inplace= True, axis =1)


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
        
    print(dataset.dtypes)
    print(dataset.head(5))
    print(dataset.tail(5))

    return dataset ,target

#Data needs reshaing as RNN requires sequence data
#in RNN, GRU and LSTM, input is of shape t, where N is the number of samples, T is length of time sequence and D is the number of features.
#so input(shape = (D,)) for ANN and Input(shape = (T,D)
def shape_data(training_data,time_step,columns):
    shaped_data = []
    for index,_ in training_data.iterrows():
        
        #append next timestep records
        temp = training_data.iloc[index:index+time_step,:].to_numpy() #no need for index-1 as indexes are reset
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
def main(symbol,update = False,plot= False,epochs = 150, batch_size =10,time_step = 7,loss_function = 'mse',optimiser = 'adam',learning_rate = 0.001,layer_size = 32, activation = 'tanh',dropout = 0.1):
    data = query("SELECT * FROM DAILY WHERE Symbol = '{}' ORDER BY DATE ASC".format(symbol))


    #remove un-needed columns
    #ticker_data.drop(labels=['Symbol','Date','fiscalDateEnding'],axis = 1,inplace =True)

    #prepare and clean the data
    dataset , target = prepare_data(data)
    columns = dataset.shape[-1] #the size of the datframe's columns after redundent columns have been removed

    #Reshape Data
    x = shape_data(dataset,time_step,columns) 
    target = target.iloc[0:-time_step,] #account for reshaping take of last timstep records from the targets

    #splitdata
    y_test , y_train = split_data(target) 
    x_test , x_train = split_data(x)    

    model = build_model((time_step,columns),head_size =64,num_heads = 4,ff_dim = 4,num_transformer_blocks = 4,mlp_units =[64,64,64],mlp_dropout=0.4,dropout=0.2)

    model.compile(
    loss="mse",
    optimizer='adam',
    metrics=['mae','mse','mape'],
    )

    model.summary()

    model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size)

    print('==================================')
    print('           Evaluation')
    print('----------------------------------')
    results = model.evaluate(x_test , y_test)
    
    train_price = model.predict(x_train)
    test_price = model.predict(x_test)

    if plot:

        plt.plot(range(0,x.shape[0]),target, label = 'Actual Price',color = 'green' )
        plt.plot(range(0,x_train.shape[0]),train_price, label = 'Train Price',color = 'blue')
        plt.plot(range(x_train.shape[0],(x_train.shape[0] + x_test.shape[0])),test_price, label = 'Test Price', color='yellow')
        plt.legend()
        plt.title('Transformer predictions for {}'.format(symbol))
        plt.ylabel('Daily Change in Price $ (USD)')
        plt.show()

    #model.save('Transformer_{}'.format(symbol))

    return model,results

if __name__ == '__main__':
    main('AAPL',update = False ,plot = True,epochs = 150,time_step = 7,batch_size = 10,activation = 'tanh')