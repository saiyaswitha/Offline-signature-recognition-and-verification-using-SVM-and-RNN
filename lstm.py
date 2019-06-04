import numpy as np
from numpy.random import seed
seed(1)
import tensorflow as tf
from tensorflow import set_random_seed
set_random_seed(2)
import pandas as pd
from time import time
from tensorflow.python.ops import rnn, rnn_cell
from sklearn.preprocessing import QuantileTransformer

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_input = 369
#train_person_id = input("Enter person's id : ")
#test_image_path = input("Enter path of signature image : ")

train_path = 'trainingbin_.csv'
#test.testing(test_image_path)
#test_path = 'test.csv'

def readCSV(train_path, type2=False):
    # Reading train data
    #df = pd.read_csv(train_path, usecols=range(n_input))
    #train_input = np.array(df.values)
    #train_input = train_input.astype(np.float32, copy=False)  # Converting input to float_32
    #df = pd.read_csv(train_path, usecols=(n_input,))
    #temp = [elem[0] for elem in df.values]
    bankdata = pd.read_csv(train_path)
    x1 = bankdata.drop('class_label', axis=1)  
    y1 = bankdata['class_label']
    from sklearn.model_selection import train_test_split  
    epoch_x, test_x, epoch_y, test_y = train_test_split(x1, y1, test_size = 0.25) 
    epoch_x = np.array(epoch_x.values)
    epoch_x = epoch_x.astype(np.float32, copy=False) # Converting input to float_32
    #print(epoch_x)
    test_x = np.array(test_x.values)
    test_x = test_x.astype(np.float32, copy=False)  
    #print(temp)
    #print('ddddddddddddddddddddddddddd')
    correct = np.array(epoch_y.values)
    #print(correct)
    corr_train = np.eye(55)[correct]
    correct1 = np.array(test_y.values)
    corr_test = np.eye(55)[correct1]
    #print(corr_test)
    '''# Converting to one hot
    # Reading test data
    #print(corr_train)
    df = pd.read_csv(test_path, usecols=range(n_input))
    test_input = np.array(df.values)
    test_input = test_input.astype(np.float32, copy=False)
    if not(type2):
        df = pd.read_csv(test_path, usecols=(n_input,))
        temp = [elem[0] for elem in df.values]
        correct = np.array(temp)
        corr_test = np.eye(55)[correct]      # Converting to one hot'''
    if not(type2):
        return epoch_x, corr_train, test_x, corr_test
    else:
        return epoch_x, corr_train, test_x


hm_epochs = 500
n_classes = 55
batch_size = 18
chunk_size = 123
n_chunks = 3
rnn_size = 440

x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes], seed=1)),
             'biases':tf.Variable(tf.random_normal([n_classes], seed=2))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output

def train_neural_network(x):

    logits = recurrent_neural_network(x)
    prediction = tf.nn.softmax(logits)
    #prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        #from sklearn.model_selection import train_test_split  
        #epoch_x, test_x, epoch_y, test_y = train_test_split(x1, y1, test_size = 0.25) 
        #print(epoch_x.shape)
        epoch_x1, epoch_y1, test_x1, test_y1 = readCSV(train_path)
        sc = QuantileTransformer(output_distribution='uniform')
        epoch_x1 = sc.fit_transform(epoch_x1)
        epoch_y1 = sc.fit_transform(epoch_y1)
        test_x1 = sc.fit_transform(test_x1)
        test_y1 = sc.fit_transform(test_y1)
        epoch_x1=np.split(epoch_x1,55)
        #print(epoch_x1)
        epoch_y1=np.split(epoch_y1,55)
        for epoch in range(hm_epochs):
            epoch_loss = 0
            #epoch_y=np.split(epoch_y,20)
            for i,j in zip(epoch_x1,epoch_y1):
                e_x = i
                e_y=j
                e_x = e_x.reshape((batch_size,n_chunks,chunk_size))
                #e_x = .reshape(e_x, shape=[batch_size,n_chunks,chunk_size])

                _, c = sess.run([optimizer, cost], feed_dict={x: e_x, y: e_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x1.reshape((330, n_chunks, chunk_size)), y:test_y1}))
        '''pred = prediction.eval({x: test_x})
        print(pred)
        x=np.argmax(pred[0])

        if x<55:
            print('Genuine Image')
            print(x)
            return True
        else:
            print('Forged Image')
            print(x)
            return False'''

train_neural_network(x)