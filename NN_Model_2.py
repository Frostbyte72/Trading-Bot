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
    training_data = pd.DataFrame()
    #set intial date in datetime datatype
    length = len(technical)-1 #-1 to account for zero indexing
    report_date = datetime.strptime(fundemental['fiscalDateEnding'].iloc[0], '%Y-%m-%d')
    for i,row in technical.iterrows():
        row_date = datetime.strptime(row['Date'], '%Y-%m-%d %X')

        #create the Target Value for the row
        row['Target'] = technical['Close'].iloc[length-i-1]

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
        training_data = training_data.append(row,ignore_index=True)
    
        

    ########################## needs updating to split the data frame into 2 parts one for testing one for training i.e 80% train 20%test
    
    #fraction of training data to be used for testing
    """test_split = 0.1
    test_data = training_data.iloc[training_data.shape[0]-int(training_data.shape[0] * test_split):-1,:]
    traning_data = training_data.iloc[0:training_data.shape[0]-int(training_data.shape[0] * test_split)]"""
    
    #removes the frist row for the last known day of trading as we don't yet know tommorows close so it can have no taget value and thus useless training
    training_data.drop(index = 0,axis=0,inplace=True)


    print(training_data.head(5))
    return training_data

# --- Prepares the data for the neural network ---
# Cleans data by removing null values and data that dosn't change
# also scales the data using the min max range to ensure that for any datapoint x in the data set is -1<=x<=1
def prepare_data(dataset):
    for col in list(dataset.columns):
        dataset[str(col)] = dataset[str(col)].fillna(0)
        if dataset[str(col)].all() == dataset[str(col)].iloc[0]: #if all values are the same remove
            dataset.drop(labels=str(col),axis=1,inplace=True)
            continue


                #Normalise the Data
        min = dataset[str(col)].min()

        #if min is <0 add i add the absoloute value of the min to all records so you only have poistive values whilst keeping the scale.
        if min <0:
            dataset[str(col)] = dataset[str(col)].apply(lambda x: x + abs(min))
            print('herro' + str(col))
        
        max = dataset[str(col)].max()
        min = dataset[str(col)].min() #needs recalculating as it may have changed

        dataset[str(col)] = dataset[str(col)].apply(lambda x: (x-abs(min))/(abs(max)-abs(min)))

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

def create_model(shape,loss_function,optimiser,learning_rate,layer_size,activation,dropout):
    
    model = Sequential()
    
    model.add(layers.LSTM(layer_size, input_shape=(shape), name = 'input',return_sequences=True )) #setsup first hidden layer aswell as defing the input layers shape
    model.add(layers.LSTM(layer_size, activation=activation, name = 'Layer2',return_sequences=True))
    model.add(layers.Dropout(rate=dropout)) #Dropout later removes records at the given rate i.e rate=0.2 1/5 records will randomly be dropped in each epoch to reduce overfitting of the data
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
    
def main(symbol,explain):
    training_data = organise_data(symbol)
    time_step = 60 #time step of 7 will mean that the last rolling weeks data is used to predict the price
    
    #Create Target
    target = training_data['Target']
    target = target.iloc[0:-time_step]

    #Remove unessecary columns
    training_data.drop(labels=['Target','Symbol','Date','fiscalDateEnding'],axis = 1,inplace =True) #Dropping open,close,high annd low resulted in lower loss (I think because they are so close to the price they get weighted heavily when they don't rly effect the next price that much history dosn't equlat the future.)

    #Manually remove Harmful features just creating noise for GOOGL
    training_data.drop(labels=['profitMargin','changeInCashAndCashEquivalents','profitLoss','operatingCashflow','totalCurrentAssets'],axis = 1, inplace = True)

    #prepare the data
    test_record = training_data.iloc[0]
    training_data = prepare_data(training_data)

    print(len(training_data.columns))
    columns = training_data.shape[-1] #the size of the datframe's columns after redundent columns have been removed


    #Reshape Data
    x_train = shape_data(training_data,time_step,columns) 
    
    #shape,loss_function,optimiser,learning_rate,layer_size,activation,dropout
    model = create_model((time_step,columns),'mse','adam',0.001,32,'tanh',0.2)

    model.fit(x_train, target, epochs = 150, batch_size = 10)

    print('==================================')
    print('           Evaluation')
    print('----------------------------------')
    results = model.evaluate(x_train, target)
    print('Prediction of the first record :')
    test = np.reshape(x_train[0],(1,time_step,columns))
    print(test_record)
    print("Predicted Price for next day's close: ")
    print(model.predict(test))


    #Save model 
    """
    x = ''
    while x != 'y' or x != 'n':
        x = input("Save Model (y/n): ")
        if x == 'y':
            print('Model Saved')
            model.save('LSTM_Model')
        elif x == 'n':
            break """


    
    if explain:
        print('Running Feature Explainer')
        importance = explainer(target,x_train,list(training_data.columns))
        print(importance)
        #importance['Loss'].apply(lambda x: x - results[0])
        importance = importance.assign(value = lambda x:(x['Loss'] - results[0]))

        plt.barh(list(importance.index.values),importance['value'].tolist(),color ='maroon')
        plt.xlabel('Change to MSE loss')
        plt.title('Feature Importance')
        plt.show()

    return model

if __name__ == '__main__':
    explain = False

    x = ''
    while x != 'y' or x != 'n':
            x = input("Explain Feature Importance's ? (y/n): ")
            if x == 'y':
                print('Features will be explianed')
                explain = True
                break
            elif x == 'n':
                break

    main('GOOGL',explain)
