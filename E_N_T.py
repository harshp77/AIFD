import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from itertools import chain
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder



class Prep():

    def read_and_prep_data(path):
        df = pd.read_excel(path)
        X = df.iloc[:,7:28]
        label = df.iloc[:,-1:]
        removables = []
        for title in X.columns:
            removables.append(list(X[X[title]=="I/O Timeout"].index))
            removables.append(list(X[X[title]=="Bad"].index))
        removables = list(set(list(chain.from_iterable(np.array(removables)))))
        feat = X.drop(removables,axis=0)
        label = label.drop(removables,axis=0)
        label["CONDITION"] = label["CONDITION"].str.replace('NORMAL','0')
        label["CONDITION"] = label["CONDITION"].str.replace('ROTOR FAILURE','1')
        label["CONDITION"] = pd.to_numeric(label["CONDITION"])
        senorname=feat.keys()[:-1]
        for ele in feat:
            feat[ele] = pd.to_numeric(feat[ele])
        comb = pd.concat([feat,label],axis=1)
        return feat,label,senorname,comb,df

    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, namen = list(),list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            namen +=[('sensor%d(t-%d)' %(j+1, i)) for j in range (n_vars)]
            #forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                namen +=[('sensor%d(t)' %(j+1)) for j in range (n_vars)]
            else:
                namen +=[('sensor%d(t+%d)' '%'(j+1, i)) for j in range (n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns=namen
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def remove_shifted(data_win,Values):
        to_remove_list =['sensor'+str(n)+'(t)' for n in range(1,len(Values.columns)+1)] #now remove all non shifted elements again. so we retreive elements and shifted target
        data_y=data_win.iloc[:,-1] #Get the target data out before removing unwanted data
        data_x=data_win.drop(to_remove_list, axis=1) #remove sensors(t)
        data_x.drop(data_x.columns[len(data_x.columns)-1], axis=1, inplace=True)# remove target(t-n)
        return data_x,data_y


class Train():
    def splitting_and_shape_data(data_x,data_y,train_start,train_end,test_start,test_end):    
        train_X=data_x[train_start*1000:train_end*1000].values
        train_Y=data_y[train_start*1000:train_end*1000].values
        
        test_X=data_x[test_start*1000:test_end*1000].values
        test_Y=data_y[test_start*1000:test_end*1000].values
        
        val_X=data_x[13000:15000].values
        val_Y=data_y[13000:15000].values
        
        train_X.astype('float32')
        val_X.astype('float32')
        test_X.astype('float32')
        
        return train_X,train_Y,val_X,val_Y,test_X,test_Y

    def one_hot(train_Y,val_Y,test_Y):    
        
        oneHot=OneHotEncoder()
        oneHot.fit(train_Y.reshape(-1,1))

        train_Y_Hot =oneHot.transform(train_Y.reshape(-1,1)).toarray()
        val_Y_Hot  =oneHot.transform(val_Y.reshape(-1,1)).toarray()
        test_Y_Hot =oneHot.transform(test_Y.reshape(-1,1)).toarray()
        
        return train_Y_Hot,val_Y_Hot,test_Y_Hot  


    def reshape_for_Lstm(data):    
        # reshape for input 
        timesteps=1
        samples=int(np.floor(data.shape[0]/timesteps))

        data=data.reshape((samples,timesteps,data.shape[1]))   #samples, timesteps, sensors     
        return data



    def normalize_and_prepfor_lstm(train_X,val_X,test_X):
        timesteps=1

        scaler=MinMaxScaler().fit(train_X)
        train_X=scaler.transform(train_X) 
        samples=int(np.floor(train_X.shape[0]/timesteps))
        train_X=train_X.reshape((samples,timesteps,train_X.shape[1]))   #samples, timesteps, sensors

        scaler=MinMaxScaler().fit(val_X)
        val_X=scaler.transform(val_X) 
        samples=int(np.floor(val_X.shape[0]/timesteps))
        val_X=val_X.reshape((samples,timesteps,val_X.shape[1]))   #samples, timesteps, sensors

        scaler=MinMaxScaler().fit(test_X)
        test_X=scaler.transform(test_X)
        samples=int(np.floor(test_X.shape[0]/timesteps))
        test_X = test_X.reshape((samples,timesteps,test_X.shape[1]))   #samples, timesteps, sensors  

        return train_X,val_X,test_X


    def model_setup_Fapi(in_shape):
        from tensorflow.keras.layers import LSTM
        from tensorflow.keras.layers import Dropout
        from tensorflow.keras.layers import Dense
        
        inputs= tf.keras.Input(shape=(in_shape[1],in_shape[2]))
        x=LSTM(42,activation='relu', input_shape=(in_shape[1],in_shape[2]),return_sequences=True)(inputs)
        x=LSTM(42,activation='relu')(x)
        out_signal=Dense(1, name='signal_out')(x)
        out_class=Dense(2,activation='softmax', name='class_out')(x)
        
        model=tf.keras.Model(inputs=inputs, outputs=[out_signal,out_class])
        
        model.compile(loss={'signal_out':'mean_squared_error',
                            'class_out' :'categorical_crossentropy'},
                            optimizer='adam',
                            metrics={'class_out':'acc'})
        
        print(model.summary())
        return model

    def plot_training(history,what='loss',saving=False,name='training'):
        fig=plt.figure()
        plt.plot(history[0])
        plt.plot(history[1])
        plt.xlabel('epoch')
        plt.legend(['train', 'test'])
        if what=='loss':
            plt.title('model loss')
            plt.ylabel('loss')
        elif what=='acc':   
            plt.title('model Acc')
            plt.ylabel('Accuracy')   
        if saving==True:
            fig.savefig( name +'_'+ what + '.png', format='png', dpi=300, transparent=True)


        plt.xlabel('epoch')
        plt.legend(['train', 'test'])
        if saving==True:
            fig.savefig( name +'_ACC.png', format='png', dpi=300, transparent=True)  
        plt.show()

