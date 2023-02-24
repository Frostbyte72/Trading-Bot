import pandas as pd
import sqlite3
from calculate_indicators import query as query
from calculate_indicators import main as m
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def organise_data(symbol):
    
    fundemental = query('SELECT * FROM FUNDEMENTAL WHERE Symbol = "{}" ORDER BY fiscalDateEnding DESC'.format(symbol))
    fundemental.drop(labels='Symbol',axis=1,inplace=True)
    #Technical analysis needs storing in db
    #technical = query("SELECT * FROM DAILY ORDER BY DATE")
    technical = m()
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

    #removes the frist row for the last known day of trading as we don't yet know tommorows close so it can have no taget value and thus useless training
    #training_data.drop(index = 0,axis=0,inplace=True)

    #Removes all columns that have null values or are all zero
    for col in list(training_data.columns):
        if training_data[str(col)].isnull().all() or training_data[str(col)].all() == 0:
            training_data.drop(labels=str(col),axis=1,inplace=True)
            

    ########################## needs updating to split the data frame into 2 parts one for testing one for training i.e 80% train 20%test


    return training_data

def create_model(training_data):
    target = training_data['Target']
    training_data.drop(labels=['Symbol','Date','fiscalDateEnding','Target'],axis = 1,inplace =True)
    
    
    dimensions = training_data.shape[-1] #34 for google
    print(dimensions)
    

    model = Sequential()
    #input dim specifies the shape of the input as the data is only one dimension dim can be used in stead of shape input_shape(dimensions,) is the same
    model.add(Dense(128, input_shape=(dimensions,), activation='relu', name = 'input')) #setsup first hidden layer aswell as defing the input layers shape
    model.add(Dense(128, activation='relu', name = 'Layer2'))
    model.add(Dense(64,activation = 'relu', name='Layer3')) #another hidden layer aslo using the relu function as it is faster than the sigmoid
    model.add(Dense(1, name = 'output'))

    #Compile defines the loss function, the optimizer and the metrics. That's all.
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse', 'mae', 'mape'] )


    print(training_data.head(5))
    print(len(training_data))
    print(target)
    
    model.fit(training_data,target, epochs = 250, batch_size = 10)

    return

def main():
    symbol = 'GOOGL'
    training_data = organise_data(symbol)
    print(training_data['Target'].head(10))

    create_model(training_data)



if __name__ == '__main__':
    main()