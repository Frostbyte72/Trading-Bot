import pandas as pd
import numpy as np
import sqlite3
from calculate_indicators import query as query
from calculate_indicators import main as m
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tqdm import tqdm
import matplotlib.pyplot as plt


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
        # Convert prices to daiy change in price 
        ############################## increases the stationarity 

        row['Target'] = ( technical['Close'].iloc[length-i-1] - row['Close'])
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



    print(dataset.head(5))
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

    #drop last record as the value will be null
    dataset.drop(index=len(dataset)-1,axis=0,inplace = True)
    dataset.drop(labels=['Close','Open','Low','High','Volume'],inplace= True, axis =1) 
    
    ########################
    # Z score outlier removal
    mean = dataset['ChangeIn_Close'].mean()
    std = dataset['ChangeIn_Close'].std()
    dataset['Zscore'] = dataset['ChangeIn_Close'].apply(lambda x : (x-mean)/std) 
    print(dataset['Zscore'])
    print('3X Standard Deviation {}'.format(std*3))
    dataset = dataset[ (dataset['Zscore'] < 3) & (dataset['Zscore'] > -3 )]
    print(dataset['Zscore'].max())
    print(dataset['Zscore'].min())
    dataset.drop(labels = 'Zscore',axis = 1, inplace = True)
    #reset index as records are removed in the middle
    dataset.reset_index(inplace = True)
    

    #Create Target
    
    dataset['Target'] = dataset['ChangeIn_Close'].shift(1)
    print(dataset[['Target','ChangeIn_Close']])
    #removes the frist row for the last known day of trading as we don't yet know tommorows close so it can have no taget value and thus useless training
    dataset.drop(index = 0,axis=0,inplace=True)
    target = dataset['Target']
    dataset.drop(labels='Target',axis =1, inplace = True)



    # Remove useless feature
    for col in dataset.columns:
        dataset[col].fillna(method = 'backfill',inplace = True) #Replace unkown variables with the last known observation
        if dataset[col].max() == dataset[col].min() or dataset[col].isnull().values.any() == True: #if all values are the same remove or any null
            dataset.drop(labels=col,axis=1,inplace=True)
            print('dropping {}'.format(col))
            continue
        
        #Normalise the Data using min max normalisation
        min = dataset[col].min()
        

        #if min is <0 add i add the absoloute value of the min to all records so you only have poistive values whilst keeping the scale.
        if min <0:
            dataset[col] = dataset[col].apply(lambda x: x + abs(min))
        
        max = dataset[col].max()
        min = dataset[col].min() #needs recalculating as it may have changed
        
        dataset[col] = dataset[col].apply(lambda x: (x-abs(min))/(abs(max)-abs(min)))
        
    print(dataset.head(5))

    

    return dataset, target

#Data needs reshaing as RNN requires sequence data
#in RNN, GRU and LSTM, input is of shape t, where N is the number of samples, T is length of time sequence and D is the number of features.
#so input(shape = (D,)) for ANN and Input(shape = (T,D)
def shape_data(training_data,time_step,columns):
    shaped_data = []
    for index,_ in training_data.iterrows():
        #append next timestep records
        temp = training_data.iloc[index:index+time_step,:].to_numpy()
        shaped_data.append(temp)
        
    #remove the last time_step elements from the lsit as they won't be the correct shape.
    
    shaped_data = shaped_data[:-time_step]
    shaped_data = np.reshape(shaped_data,(len(shaped_data),time_step,columns))
    print("Reshaped Data")
    print(len(shaped_data))

    return shaped_data

def create_model(shape,loss_function,optimiser,learning_rate,layer_size,activation,dropout):
    
    model = Sequential()
    
    model.add(layers.LSTM(layer_size, input_shape=(shape), name = 'input',return_sequences=True )) #setsup first hidden layer aswell as defing the input layers shape
    model.add(layers.Dropout(rate=dropout)) 
    model.add(layers.LSTM(layer_size, activation=activation, name = 'Layer2',return_sequences=True))
    model.add(layers.Dropout(rate=dropout)) #Dropout layer removes records at the given rate i.e rate=0.2 1/5 records will randomly be dropped in each epoch to reduce overfitting of the data
    model.add(layers.LSTM(layer_size,activation = activation, name='Layer3')) #another hidden layer
    #model.add(layers.Dense(layer_size, activation=activation, name = 'Layer4'))
    model.add(layers.Dense(1,activation="linear", name = 'output')) #Output layer linear(x) = x

    #Compile defines the loss function, the optimizer and the metrics. That's all.
    model.compile(loss=loss_function, optimizer=optimiser, metrics=['mse', 'mae', 'mape'] )

    model.summary()

    return model

#shapley addative explinations
#importance of feature A = loss(A,B,C) - loss(B,C)
#https://towardsdatascience.com/a-novel-approach-to-feature-importance-shapley-additive-explanations-d18af30fc21b
def explainer(target,x_train,features):
    print(len(features))

    importance = pd.DataFrame(columns = ['Loss','mse','mae','mape'])
    #Create a new model minus one feature
    model = create_model((x_train.shape[1],x_train.shape[2]-1),'mse','adam',0.001,32,'tanh',0.2)
    inital_weights = model.get_weights()
    #Run the trails for each - one of the features
    for index in tqdm(range(0,len(features)),desc = 'Calcualting Feature Importance' ):
        #remove feature from time series

        x_train_adjusted = np.delete(x_train,index,2)
        model.fit(x_train_adjusted,target, epochs = 150, batch_size = 10,verbose = 0)
        results = model.evaluate(x_train_adjusted, target,verbose = 0)
        importance.loc[features[index]] = results

        #reset the mdoel to the same initial weights
        model.set_weights(inital_weights)

    return importance

#Arguments Decscirptions:
#-----------------------
# Symbol = String - Ticker of the stock/fund (List avalible at: https://www.alphavantage.co/documentation/)
# explain = Boolean - explain the features importance of the model
# plot = Boolean - Plot the Models prediction against actual values
# epochs = int - No. of epochs
# batch_size  = int - size of batch
# time_step = int - how many previous days tradying are provided to the LSTM model (time step of 7 will mean that the last rolling weeks data is used to predict the price)
# All other arguments are optional arguments for the keras model creation 
def main(symbol,explain = False,plot= False,epochs = 150, batch_size =10,time_step = 7,loss_function = 'mse',optimiser = 'adam',learning_rate = 0.001,layer_size = 32, activation = 'tanh',dropout = 0.1):
    dataset = organise_data(symbol)
    

    #Remove unessecary columns
    dataset.drop(labels=['Symbol','Date','fiscalDateEnding'],axis = 1,inplace =True) #Dropping open,close,high annd low resulted in lower loss (I think because they are so close to the price they get weighted heavily when they don't rly effect the next price that much history dosn't equlat the future.)

    #Manually remove Harmful features just creating noise for GOOGL
    #dataset.drop(labels=['profitMargin','changeInCashAndCashEquivalents','profitLoss','operatingCashflow','totalCurrentAssets'],axis = 1, inplace = True)

    #prepare and clean the data
    dataset,target = prepare_data(dataset)
    target = target.iloc[0:-(time_step)] #accounts for timestep
    
    

    print(len(dataset.columns))
    print(dataset.shape[0])
    columns = dataset.shape[-1] #the size of the datframe's columns after redundent columns have been removed

    #Reshape Data
    x = shape_data(dataset,time_step,columns) 
    print(target.tail(20))

    #splitdata
    x_test , x_train = split_data(x)
    y_test , y_train = split_data(target) 

    #shape,loss_function,optimiser,learning_rate,layer_size,activation,dropout
    model = create_model((time_step,columns),loss_function,optimiser,learning_rate,layer_size,activation,dropout)

    model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size)

    print('==================================')
    print('           Evaluation')
    print('----------------------------------')
    results = model.evaluate(x_test , y_test)

    train_price = model.predict(x_train)
    test_price = model.predict(x_test)



    if plot:
        #add plot of crossvalidated predictions
        plt.plot(range(0,dataset.shape[0]-time_step),target, label = 'Actual Price' )
        plt.plot(range(0,x_train.shape[0]),train_price, label = 'Train Price')
        plt.plot(range(x_train.shape[0],(x_train.shape[0] + x_test.shape[0])),test_price, label = 'Test Price')
        plt.title('LSTM Predictions for {}'.format(symbol))
        plt.show()
    
    if explain:
        print('Running Feature Explainer')

        importance = explainer(y_train,x_train,list(dataset.columns))
        print(importance)
        #importance['Loss'].apply(lambda x: x - results[0])
        importance = importance.assign(value = lambda x:(x['Loss'] - results[0]))

        plt.barh(list(importance.index.values),importance['value'].tolist(),color ='maroon')
        plt.xlabel('Loss (MSE)')
        plt.title('Feature Importance for {}'.format(symbol))
        plt.show()

    return results

if __name__ == '__main__':
    main('AAPL',plot = True,epochs = 15,batch_size = 10,layer_size = 64,activation = 'sigmoid')

