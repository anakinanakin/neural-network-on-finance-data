import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymongo as mg

from datetime import datetime, timedelta
from tensorflow.python.keras.utils import to_categorical



def date_range(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def main():
    client = mg.MongoClient('your mongodb client')
    db = client['twse']
    col = db['tw50']
    code = '2330'

    # train
    # start_date = datetime(2019, 1, 1)
    # end_date = datetime(2019, 5, 1)

    # validation
    start_date = datetime(2019, 5, 1)
    end_date = datetime(2019, 6, 1)

    data_x = []
    data_y = []

    window_size = 100

    #k=50 or 100 is good
    pred_k = 100

    #threshold=0.0010 is good
    label_threshold = 0.0010

    for dt in date_range(start_date, end_date):
        print "starting date: "+dt.strftime('%Y_%m_%d')

        col = db['tw50']
        docs = col.find( { 'Code': code, 'DspDatetime': { '$gte': dt, '$lt':  dt+timedelta(days=1) } } )

        docs = [doc for doc in docs]

        if docs == []:
            continue

        #print(len(docs))

        midprice_list = []

        # unstable before 400
        docs = docs[400:]

        #print(len(docs))

        input_data = []
        input_tensor = []

        for doc in docs:
            newdoc = []
            buypv5 = np.array(doc['BuyPV5']).flatten().tolist()
            sellpv5 = np.array(doc['SellPV5']).flatten().tolist()
            midprice = (buypv5[0]+sellpv5[0])/2
            midprice_list += [midprice]

            #print(buypv5+sellpv5, doc['Trade_Price'], midprice)

            # arrange tensor
            for i in range(5):
                newdoc += [buypv5[i*2]]
                newdoc += [buypv5[i*2+1]]
                newdoc += [sellpv5[i*2]]
                newdoc += [sellpv5[i*2+1]]

            #print newdoc
            input_data += [newdoc]

        #print len(input_data)
        # print input_data

        input_shape = len(input_data)-(window_size-1+pred_k)

        for i in range(input_shape):
            input_tensor += [input_data[i:i+window_size]]

        #print len(input_tensor)
        #print input_tensor
            

        # unstable before 400
        #midprice_list = midprice_list[400:]

        df = pd.DataFrame(midprice_list, columns=['Mid price'])


        df['horizon avg'] = 0.0

        #list slicing doesn't include last element; pd.Dataframe loc does include
        for i in df.index:
            df.loc[i,'horizon avg'] = df.loc[i+1:i+pred_k]['Mid price'].sum()/float(pred_k)


        df['pct'] = (df['horizon avg']-df['Mid price'])/df['Mid price']


        #for train_x, train_y
        df['target'] = 1

        #labels 0: equal or greater than label_threshold
        #labels 1: between
        #labels 2: smaller or equal to -label_threshold
        df.loc[df['pct'] >=       label_threshold, 'target'] = 0
        df.loc[df['pct'] <=  (-1)*label_threshold, 'target'] = 2

        label = df['target'].values.tolist()
        #print(len(label))
        label = label[window_size-1:-pred_k]
        #print(len(label))

        data_x += input_tensor
        data_y += label

    #print len(data_x)
    #print len(data_y)

    data_x = np.array(data_x)
    data_y = np.array(data_y)

    data_x = data_x.reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2], 1)
    data_y = data_y.astype(int)

    #print data_x.shape
    #print data_y.shape

    save_dir = os.path.join(os.getcwd(), 'data_set')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # np.save(os.path.join(save_dir, 'train_x.npy'), data_x)
    np.save(os.path.join(save_dir, 'valid_x.npy'), data_x)

    data_y = to_categorical(data_y)
    # np.save(os.path.join(save_dir, 'train_y_onehot.npy'), data_y)
    np.save(os.path.join(save_dir, 'valid_y_onehot.npy'), data_y)




if __name__ == '__main__':
    main()