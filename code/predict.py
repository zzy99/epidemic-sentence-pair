import numpy as np
import pandas as pd
import os
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from bert4keras.backend import keras, K
from bert4keras.bert import build_bert_model
from bert4keras.tokenizer import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding
from sklearn.model_selection import StratifiedKFold
import Levenshtein
import datetime

np.random.seed(2020)

train_df = pd.read_csv('../data/Dataset/train.csv')
valid_df = pd.read_csv('../data/Dataset/dev.csv')
test_df = pd.read_csv('../data/Dataset/test.csv')
train_ext_df = pd.read_csv('../data/External/chip2019.csv')

train_df.dropna(axis=0,inplace=True)

train_data = train_df[['query1','query2','label']].values
valid_data = valid_df[['query1','query2','label']].values
test_data = test_df[['query1','query2','label']].values
train_ext_data = train_ext_df[['question1','question2','label']].values

def build_model(mode='bert', filename='bert', lastfour=False, LR=1e-5, DR=0.2):
    path = '../data/External/'+filename+'/'
    config_path = path+'bert_config.json'
    checkpoint_path = path+'bert_model.ckpt'
    dict_path = path+'vocab.txt'

    global tokenizer
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    bert = build_bert_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        with_pool=True,
        model=mode,
        return_keras_model=False,
    )
    if lastfour:
        model = Model(
            inputs=bert.model.input,
            outputs=[
                bert.model.layers[-3].get_output_at(0),
                bert.model.layers[-11].get_output_at(0),
                bert.model.layers[-19].get_output_at(0),
                bert.model.layers[-27].get_output_at(0),
            ]
        )
        output = model.outputs
        output1 = Lambda(lambda x: x[:, 0], name='Pooler1')(output[0])
        output2 = Lambda(lambda x: x[:, 0], name='Pooler2')(output[1])
        output3 = Lambda(lambda x: x[:, 0], name='Pooler3')(output[2])
        output4 = Lambda(lambda x: x[:, 0], name='Pooler4')(output[3])

        output = Concatenate(axis=1)([output1, output2, output3, output4])

    else:
        output = bert.model.output

    output = Dropout(rate=DR)(output)
    output = Dense(units=2,
                   activation='softmax',
                   kernel_initializer=bert.initializer)(output)

    model = Model(bert.model.input, output)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(LR),
        metrics=['accuracy'],
    )
    return model

class data_generator(object):
    def __init__(self, data, batch_size=32, random=True):
        self.data = data
        self.batch_size = batch_size
        self.random = random
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            text1, text2, label = self.data[i]
            token_ids, segment_ids = tokenizer.encode(text1, text2, max_length=64)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    def forfit(self):
        while True:
            for d in self.__iter__(self.random):
                yield d


batch_size = 16

train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size, random=False)


def f(x):
    res=1/(1+np.e**(-x))
    return res

def f_ver(x):
    res=np.log(x/(1-x))
    return res

def do_predict(fileDir):
    res = []
    fileList = os.listdir(fileDir)
    for file in fileList:
        model.load_weights(fileDir+'/'+file)
        print('predicting on '+fileDir+'/'+file)
        pred = model.predict_generator(test_generator.forfit(), steps=len(test_generator))
        res.append(pred)
    s = np.zeros_like(res[0])#(N,2)
    for i in res:
        s += f_ver(i)/len(res)
    s = f(s)
    s = s[:,1]
    return s


model = build_model(mode='bert',filename='bert',lastfour=False)
res1 = do_predict('../user_data/model_data/bert_weights')

model = build_model(mode='bert',filename='ernie',lastfour=False)
res2 = do_predict('../user_data/model_data/ernie_weights')


model = build_model(mode='bert',filename='roberta',lastfour=False)
res3 = do_predict('../user_data/model_data/roberta_weights')


res = f((2.025*f_ver(res1) + 2.025*f_ver(res2) + 1.95*f_ver(res3))/6)

alpha = 0.47

test_data[:,2] = (res>=alpha).astype('int')
train_data = np.concatenate([train_data,test_data],axis=0)


model = build_model(mode='bert', filename='bert', LR=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=5,
                    validation_data=valid_generator.forfit(),
                    validation_steps=len(valid_generator),
                    callbacks=[early_stopping],
                    verbose=2,
                    )
pred_pl = model.predict_generator(test_generator.forfit(), steps=len(test_generator))
pred_pl = pred_pl[:,1]


pred = res*0.8 + pred_pl*0.2
pred = (pred>=alpha).astype('int')

for i in range(len(test_data)):
    d = test_data[i]
    texta = d[0]
    textb = d[1]
    if Levenshtein.distance(texta,textb) == 0:
        pred[i] = 1

test_df['label'] = pred
sub = test_df[['id','label']]
sub.to_csv(("../prediction_result/result_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), header=None, index=False)