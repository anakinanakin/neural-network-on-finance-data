import psycopg2, psycopg2.extras
import os
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import patches
from matplotlib.pyplot import figure
from datetime import timedelta, date
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score

#error occurs when directly import keras without tensorflow.python
from tensorflow.python.keras import layers, Input, regularizers
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils import to_categorical, model_to_dot, plot_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping



# 1. input_size=30: the most 30 recent updates
# 2. feature_num=4: 4 features(OHLC) per time stamp
# 3. pred_k=10: 10 future close price, k=10
# each day don't use target label of first s-1 rows and last k rows(s = input_size, k = pred_k)
def get_model_data(df, input_size, feature_num, pred_k):
    dt_count = df['dt'].value_counts()
    date_num = dt_count.shape[0]
    event_num = dt_count.sum()
    input_shape = event_num-(input_size-1+pred_k)
    df = df.drop(columns = ['dt'])

    data = df.values
    X = []
    Y = []
    #shape = data.shape
    #X = np.zeros((input_shape, input_size, feature_num))
    #Y = np.zeros((input_shape, 1))
    #e.g. take feature from 0~99 row to predict target label on 99th row, take feature from 31837~31936 row to predict target label on 31936th row
    for i in range(input_shape):#range = 0~31837
        X.append(data[i:i+input_size,0:feature_num])# [every 100 events from 31937 rows, take the first 40 columns as features]
        Y.append(data[i+input_size-1,-1:])# [from 99~31936 rows, take the last 5 columns as labels]
    #X = X.reshape(len(X), input_size, feature_num, 1)# add the 4th dimension: 1 channel

    return X,Y

def date_range(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

# transform array to rectangle shape
def trans2rect(arr):
    tarr = []
    trend = arr[0]
    width = 1
    day = 0
    for elm in arr[1:]:
        if elm == trend:
            width += 1
        else:
            tarr.append((trend, day, width))
            trend = elm
            day  += width
            width = 1
    tarr.append((trend, day, width))
    return tarr

# def train_model(model, train_x, train_y, test_x, test_y, batch_size=32, epochs=5, validation_split=0.2):
#     train_acc = []
#     train_loss = []
#     valid_acc = []
#     valid_loss = []
#     test_acc = []
#     test_loss = []
#     test_f1 = []
#     test_precision = []
#     test_recall = []

#     test_y_label = [np.where(r==1)[0][0] for r in test_y]

#     for i in range(epochs):
#         print('starting epoch {}'.format(i+1))
#         history1 = model.fit(train_x, train_y, batch_size=batch_size, epochs=1, validation_split=validation_split)

#         train_acc =  train_acc  + history1.history['acc']
#         train_loss = train_loss + history1.history['loss']
#         valid_acc =  valid_acc  + history1.history['val_acc']
#         valid_loss = valid_loss + history1.history['val_loss']

#         loss, acc = model.evaluate(test_x, test_y)
#         test_loss = test_loss + [loss]
#         test_acc =  test_acc + [acc]
        
#         y_pred = model.predict(test_x, verbose=1)
#         y_pred = np.argmax(y_pred, axis=1)
#         y_pred.tolist()

#         #target_names = ['increase', 'stationary', 'decrease']
#         #print(classification_report(test_y_label, y_pred, target_names=target_names))
#         f1 = f1_score(test_y_label, y_pred, average='macro')
#         precision = precision_score(test_y_label, y_pred, average='macro')
#         recall = recall_score(test_y_label, y_pred, average='macro')

#         test_f1 = test_f1 + [f1]
#         test_precision = test_precision + [precision]
#         test_recall = test_recall + [recall]

#     #print(train_acc, train_loss, valid_acc, valid_loss, test_acc, test_loss, test_f1, test_precision, test_recall)
#     return train_acc, train_loss, valid_acc, valid_loss, test_acc, test_loss, test_f1, test_precision, test_recall

def get_f1_pre_recall(model, x, y):
    y_pred = model.predict(x, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred.tolist()

    f1 = f1_score(y, y_pred, average='macro')
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')

    return f1, precision, recall


def train_model(model, save_dir, task_id, train_x, train_y, valid_x, valid_y, test_x, test_y, batch_size=32, epochs=5):

    save_dir_model = os.path.join(save_dir, 'model')
    if not os.path.isdir(save_dir_model):
        os.makedirs(save_dir_model)

    save_dir_output = os.path.join(save_dir, 'output')
    if not os.path.isdir(save_dir_output):
        os.makedirs(save_dir_output)

    train_acc = []
    valid_acc = []
    test_acc = []

    train_loss = []
    valid_loss = []
    test_loss = []

    train_f1 = []
    valid_f1 = []
    test_f1 = []

    train_precision = []
    valid_precision = []
    test_precision = []

    train_recall = []
    valid_recall = []
    test_recall = []

    train_y_label = [np.where(r==1)[0][0] for r in train_y]
    valid_y_label = [np.where(r==1)[0][0] for r in valid_y]
    test_y_label = [np.where(r==1)[0][0] for r in test_y]

    for i in range(epochs):
        print('starting epoch {}'.format(i+1))
        history1 = model.fit(train_x, train_y, batch_size=batch_size, epochs=1, validation_data=(valid_x, valid_y))

        if (i+1)%100==0:
            #model.save(os.path.join(save_dir,'model/model_epoch:{}.h5'.format(i+1)))
            model.save('result/{}/'.format(task_id)+'model/model_epoch_{}.h5'.format(i+1))

        train_acc =  train_acc  + history1.history['acc']
        train_loss = train_loss + history1.history['loss']
        valid_acc =  valid_acc  + history1.history['val_acc']
        valid_loss = valid_loss + history1.history['val_loss']

        loss, acc = model.evaluate(test_x, test_y)
        test_loss = test_loss + [loss]
        test_acc =  test_acc + [acc]

        f1, precision, recall = get_f1_pre_recall(model, train_x, train_y_label)

        train_f1 = train_f1 + [f1]
        train_precision = train_precision + [precision]
        train_recall = train_recall + [recall]

        f1, precision, recall = get_f1_pre_recall(model, valid_x, valid_y_label)

        valid_f1 = valid_f1 + [f1]
        valid_precision = valid_precision + [precision]
        valid_recall = valid_recall + [recall]

        f1, precision, recall = get_f1_pre_recall(model, test_x, test_y_label)

        test_f1 = test_f1 + [f1]
        test_precision = test_precision + [precision]
        test_recall = test_recall + [recall]

    final_train_acc = train_acc[-1]
    final_valid_acc = valid_acc[-1]
    final_test_acc = test_acc[-1]

    train_acc = np.array(train_acc)
    valid_acc = np.array(valid_acc)
    test_acc = np.array(test_acc)

    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    test_loss = np.array(test_loss)

    train_f1 = np.array(train_f1)
    valid_f1 = np.array(valid_f1)
    test_f1 = np.array(test_f1)

    train_precision = np.array(train_precision)
    valid_precision = np.array(valid_precision)
    test_precision = np.array(test_precision)

    train_recall = np.array(train_recall)
    valid_recall = np.array(valid_recall)
    test_recall = np.array(test_recall)

    np.save(os.path.join(save_dir, 'output/train_acc.npy'), train_acc)
    np.save(os.path.join(save_dir, 'output/valid_acc.npy'), valid_acc)
    np.save(os.path.join(save_dir, 'output/test_acc.npy'), test_acc)

    np.save(os.path.join(save_dir, 'output/train_loss.npy'), train_loss)
    np.save(os.path.join(save_dir, 'output/valid_loss.npy'), valid_loss)
    np.save(os.path.join(save_dir, 'output/test_loss.npy'), test_loss)

    np.save(os.path.join(save_dir, 'output/train_f1.npy'), train_f1)
    np.save(os.path.join(save_dir, 'output/valid_f1.npy'), valid_f1)
    np.save(os.path.join(save_dir, 'output/test_f1.npy'), test_f1)

    np.save(os.path.join(save_dir, 'output/train_precision.npy'), train_precision)
    np.save(os.path.join(save_dir, 'output/valid_precision.npy'), valid_precision)
    np.save(os.path.join(save_dir, 'output/test_precision.npy'), test_precision)

    np.save(os.path.join(save_dir, 'output/train_recall.npy'), train_recall)
    np.save(os.path.join(save_dir, 'output/valid_recall.npy'), valid_recall)
    np.save(os.path.join(save_dir, 'output/test_recall.npy'), test_recall)

    return final_train_acc, final_valid_acc, final_test_acc





task = pd.read_csv("task.csv") 
print(task)

output_csv = task
output_csv = output_csv.drop(columns = ['data_set'])
output_csv['train_acc'] = 0.0
output_csv['valid_acc'] = 0.0
output_csv['test_acc'] = 0.0
output_csv['completed'] = 0

print(output_csv)

for i in range(2):
    print('Running experiment {}'.format(i+1))

    task_id = int(task['task_id'][i])
    input_size = int(task['input'][i])
    pred_k = int(task['k'][i])
    feature_num = int(task['feature_num'][i])
    label_threshold = float(task['label_threshold'][i])
    lr = float(task['learning_rate'][i])
    regularizer = float(task['regularizer'][i])
    data_set = int(task['data_set'][i])

    #clear previous models
    clear_session()

    #input_tensor = Input(shape=(30,4,1))
    input_tensor = Input(shape=(input_size,feature_num,1))

    layer_x = layers.Conv2D(16, (1,4), kernel_regularizer=regularizers.l1(l=regularizer))(input_tensor)
    layer_x = layers.BatchNormalization()(layer_x)
    layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

    layer_x = layers.Conv2D(16, (4,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(layer_x)
    layer_x = layers.BatchNormalization()(layer_x)
    layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

    layer_x = layers.Conv2D(16, (4,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(layer_x)
    layer_x = layers.BatchNormalization()(layer_x)
    layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

    # Inception Module
    tower_1 = layers.Conv2D(32, (1,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(layer_x)
    tower_1 = layers.BatchNormalization()(tower_1)
    tower_1 = layers.LeakyReLU(alpha=0.01)(tower_1)
    tower_1 = layers.Conv2D(32, (3,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(tower_1)
    tower_1 = layers.BatchNormalization()(tower_1)
    tower_1 = layers.LeakyReLU(alpha=0.01)(tower_1)

    tower_2 = layers.Conv2D(32, (1,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(layer_x)
    tower_2 = layers.BatchNormalization()(tower_2)
    tower_2 = layers.LeakyReLU(alpha=0.01)(tower_2)
    tower_2 = layers.Conv2D(32, (5,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(tower_2)
    tower_2 = layers.BatchNormalization()(tower_2)
    tower_2 = layers.LeakyReLU(alpha=0.01)(tower_2)  

    tower_3 = layers.MaxPooling2D((3,1), padding='same', strides=(1,1))(layer_x)
    tower_3 = layers.Conv2D(32, (1,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(tower_3)
    tower_3 = layers.BatchNormalization()(tower_3)
    tower_3 = layers.LeakyReLU(alpha=0.01)(tower_3)

    layer_x = layers.concatenate([tower_1, tower_2, tower_3], axis=-1)

    # concatenate features of tower_1, tower_2, tower_3
    layer_x = layers.Reshape((input_size,96))(layer_x)
    #layer_x = layers.Reshape((input_size,feature_num))(input_tensor)

    # # 64 LSTM units
    #layer_x = layers.LSTM(64)(layer_x)
    layer_x = layers.LSTM(64, kernel_regularizer=regularizers.l1(l=regularizer))(layer_x)
    # # The last output layer uses a softmax activation function
    output = layers.Dense(3, activation='softmax')(layer_x)


    model = Model(input_tensor, output)
    opt = Adam(lr=lr, epsilon=1)# learning rate and epsilon are the same as paper DeepLOB 
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 
    #model.summary()




# conn = psycopg2.connect(**eval(open('auth.txt').read()))
# cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
# cmd.execute('select * from market_index where mid = 3 and dt=%(dt)s',dict(dt='2005-02-01'))
# recs = cmd.fetchall()

# df = pd.DataFrame(recs, columns = recs[0].keys())

# df['co'] = df['close']-df['open']

# #change column order
# df.loc[:,['dt', 'tm', 'open', 'close', 'high', 'low']]

# train_x = []
# train_y = []

# start_date = date(2010, 8, 1)
# end_date = date(2010, 8, 31)
# #working_day = 0
# #figure(num=None, figsize=(48, 10), dpi=80, facecolor='w', edgecolor='k')

# #run from start_date to end_date-1day
# for single_date in date_range(start_date, end_date):
#     cmd.execute('select * from market_index where mid = 3 and dt=%(dt)s',dict(dt=single_date.strftime("%Y-%m-%d")))
#     recs = cmd.fetchall()
#     #print(recs)
#     if recs == []:
#         continue;

# #     working_day = working_day+1

# # print(working_day)
#     df = pd.DataFrame(recs, columns = recs[0].keys())

#     #cmd.execute('select * from market_index where mid = 3 and dt between %(dt1)s and %(dt2)s',dict(dt1='2005-01-01', dt2='2005-01-10'))
#     #len(recs)

#     df.sort_values(by='dt')

#     #df = df[df.origin == True]

#     df = df.drop(columns = ['mid', 'tm', 'volume', 'origin'])

#     #percentage change of each row
#     #df['pct'] = df['close'].pct_change()
#     #df['pct'] = df['pct'].shift(-1)

#     df['horizon avg'] = 0.000000

#     #use previous 30mins to predict 10 min horizon(k=10)

#     #list slicing doesn't include last element; pd.Dataframe loc does include
#     for i in df.index:
#         df['horizon avg'][i] = df.loc[i+1:i+pred_k]['close'].sum()/float(pred_k)

#     df['pct'] = (df['horizon avg']-df['close'])/df['close']

#     df['target'] = 1

#     #labels 0: equal or greater than 0.00015
#     #labels 1: between
#     #labels 2: smaller or equal to -0.00015
#     df.loc[df['pct'] >=       label_threshold, 'target'] = 0
#     df.loc[df['pct'] <=  (-1)*label_threshold, 'target'] = 2

#     # label = df['target'].values.tolist()
#     # ax = plt.subplot(111)
#     # tans = trans2rect(label)

#     # tans_stats = sorted(tans, key=lambda x: x[2])
#     # for a in tans:
#     #     if a[0] == 0:
#     #         col = (1,.6,.6)
#     #     elif a[0] == 1:
#     #         col = 'w'
#     #     elif a[0] == 2:
#     #         col = (.6,1,.6) 

#     #     ax.add_patch(patches.Rectangle((a[1],0), a[2],1, color=col))

#     # close_price = df['close'].values.tolist()
#     # close_price = [(float(i)-min(close_price))/(max(close_price)-min(close_price)) for i in close_price]
#     # plt.plot(close_price)
#     # #plt.plot(ps)
#     # plt.title('date={}, k={}, threshold={}, #lables={}, max_period={}'.format(single_date, pred_k, label_threshold, len(tans), tans_stats[-1][2]))
#     # plt.savefig('date={}_k={}_threshold={}.png'.format(single_date, pred_k, label_threshold*10000))
#     # plt.clf()

#     # print(df['target'].value_counts())

#     df = df.drop(columns = ['pct', 'horizon avg'])

#     #mean = (df['open'].mean()+df['high'].mean()+df['low'].mean()+df['close'].mean())/4

#     df1 = df['open']
#     df2 = df['high']
#     df3 = df['low']
#     df4 = df['close']

#     df5 = pd.concat([df1, df2, df3, df4], ignore_index=True)
#     mean = df5.mean()
#     std = df5.std()

#     #zscore
#     df['open'] = (df['open']-mean)/std
#     df['high'] = (df['high']-mean)/std
#     df['low'] = (df['low']-mean)/std
#     df['close'] = (df['close']-mean)/std

#     #if single_date == start_date:
#     #else:
#     x, y = get_model_data(df, input_size, feature_num, pred_k)
#     #print(x)
#     #print(y)

#     #np array
#     #train_x = np.append(train_x, x, axis = 0)
#     #train_y = np.append(train_y, y, axis = 0)
#     #list
#     train_x = train_x + x
#     train_y = train_y + y

# train_x = np.array(train_x)
# train_y = np.array(train_y)
# train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)


# train_y = train_y.astype(int)

    save_dir = os.path.join(os.getcwd(), 'data_set/'+str(data_set))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    train_x = np.load(os.path.join(save_dir, 'train_x.npy'))
    train_y = np.load(os.path.join(save_dir, 'train_y_onehot.npy'))
    valid_x = np.load(os.path.join(save_dir, 'valid_x.npy'))
    valid_y = np.load(os.path.join(save_dir, 'valid_y_onehot.npy'))
    test_x = np.load(os.path.join(save_dir, 'test_x.npy'))
    test_y = np.load(os.path.join(save_dir, 'test_y_onehot.npy'))
                
# # print(train_x.shape)
# # print(train_y.shape)
# # print(valid_x)
# # print(valid_y)
# # print(test_x)
# # print(test_y)

# #np.save(os.path.join(save_dir, 'test_x_2010_07_ohlc_price_input={}_k={}_threshold={}.npy'.format(input_size, pred_k, label_threshold)), train_x)
# #np.save(os.path.join(save_dir, 'test_y_2010_07_ohlc_price_input={}_k={}_threshold={}.npy'.format(input_size, pred_k, label_threshold)), train_y)
# np.save(os.path.join(save_dir, 'test_x.npy'), train_x)
# np.save(os.path.join(save_dir, 'test_y.npy'), train_y)

# train_y = to_categorical(train_y)

# #np.save(os.path.join(save_dir, 'test_y_2010_07_ohlc_price_input={}_k={}_threshold={}_onehot.npy'.format(input_size, pred_k, label_threshold)), train_y)
# np.save(os.path.join(save_dir, 'test_y_onehot.npy'), train_y)


    # save_dir = os.path.join(os.getcwd(), 'result/'+str(task_id)+'/model')
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)

    #filepath = "model_{epoch:02d}-{val_loss:.2f}.hdf5"
    #save model for each epoch
    #checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_loss',verbose=1, period=100)

    save_dir = os.path.join(os.getcwd(), 'result/'+str(task_id))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'readme.txt'), 'w') as f:
        f.write('task_id,input,k,feature_num,label_threshold,learning_rate,regularizer,data_set\n')
        f.write(str(task_id)+',\t\t'+str(input_size)+',\t\t'+str(pred_k)+',\t\t'+str(feature_num)+',\t\t'+str(label_threshold)+',\t\t'+str(lr)+',\t\t'+str(regularizer)+',\t\t'+str(data_set))

# # # # early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1)





# history1 = model.fit(train_x, train_y, batch_size=32, epochs=500, callbacks=[checkpoint], validation_data=(valid_x, valid_y))
#history1 = model.fit(train_x, train_y, batch_size=32, epochs=5, validation_split=0.2)

#model = load_model('model_100-0.58.hdf5')

#train_acc, train_loss, valid_acc, valid_loss, test_acc, test_loss, test_f1, test_precision, test_recall = train_model(model, train_x, train_y, test_x, test_y, batch_size=32, epochs=5, validation_split=0.2)
    train_acc, valid_acc, test_acc = train_model(model, save_dir, task_id, train_x, train_y, valid_x, valid_y, test_x, test_y, batch_size=32, epochs=1000)

    #print(train_acc)
    output_csv['train_acc'][i] = train_acc
    output_csv['valid_acc'][i] = valid_acc
    output_csv['test_acc'][i] = test_acc
    output_csv['completed'][i] = 1
    output_csv.to_csv('output.csv')


# # #test_dir = os.path.join(os.getcwd(), 'test_models')
# # files = glob.glob(os.path.join(save_dir, '*.hdf5'))
# # files.sort(key=os.path.getmtime)

# # test_loss = []
# # test_acc = []


# # #use too much memory
# # #load model and draw each epoch's accuracy and loss
# # for f in files:
# #     #print(f)
# #     model = load_model(f)
# #     loss, acc = model.evaluate(test_x, test_y)
# #     test_loss = test_loss + [loss]
# #     test_acc = test_acc + [acc]



# #plot train and validation accuracy
# plt.plot(history1.history['acc'])
# plt.plot(history1.history['val_acc'])
# # plt.plot(train_acc)
# # plt.plot(valid_acc)
# #plt.plot(test_acc)
# plt.title('Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Valid'])
# # plt.legend(['Train', 'Valid', 'Test'])
# plt.savefig('accuracy.png')
# plt.clf()

# # #plot train and validation loss
# plt.plot(history1.history['loss'])
# plt.plot(history1.history['val_loss'])
# # plt.plot(train_loss)
# # plt.plot(valid_loss)
# #plt.plot(test_loss)
# plt.title('Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Valid'])
# # plt.legend(['Train', 'Valid', 'Test'])
# plt.savefig('loss.png')
# plt.clf()

# plt.plot(test_acc)
# plt.plot(test_recall)
# plt.plot(test_precision)
# plt.plot(test_f1)
# plt.title('Test evaluation')
# plt.ylabel('Score')
# plt.xlabel('Epoch')
# plt.legend(['Accuracy', 'Recall', 'Precision', 'F1'])
# plt.savefig('test_evaluation.png')
# plt.clf()


# model = load_model('model_09-1.00.hdf5')

# test_y = [np.where(r==1)[0][0] for r in test_y]

# y_pred = model.predict(test_x, verbose=1)
# y_pred = np.argmax(y_pred, axis=1)
# y_pred.tolist()
# #conf = confusion_matrix(test_y, y_pred)
# print(test_y)
# print(y_pred)
#print(conf)


# target_names = ['increase', 'stationary', 'decrease']
# print(classification_report(test_y, y_pred, target_names=target_names))


# #output directory naming

# tid = 12854

# #'00003236'
# hex_num = '{:08x}'.format(tid)

# #['00', '00', '32', '36']
# hex_num = ['{:08x}'.format(tid)[i*2:i*2+2] for i in range(4)]

# #'00/00/32/36'
# save_dir = '/'.join(['{:08x}'.format(tid)[i*2:i*2+2] for i in range(4)])





