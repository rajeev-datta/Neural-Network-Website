#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:51:28 2017

@author: BossBoJackson
"""
from __future__ import print_function
#import flask
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
#from sklearn.linear_model import LogisticRegression
#from six.moves.urllib.request import urlretrieve
#from six.moves import cPickle as pickle
#import time
from six.moves import range
#import xlwt
#import xlrd
#import math as math
##from IPython.core.debugger import Tracer;#Use Tracer()() to set a breakpoint from which you want to debug from
#from collections import OrderedDict
from tensorflow.python import debug as tf_debug

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


app = Flask(__name__)
# @ signifies a decorator - way to wrap a function and modifying its behavior

@app.route('/')

def homepage():
    return render_template("main.html")

@app.route('/ConvNet')

def ConvNet():
    return render_template("ConvNet.html")

@app.route('/ConvNetForm', methods=["POST"])

def ConvNetForm():
     #hyperparameters
    learning_rate = request.form['learning_rate']
    learning_rate = eval(learning_rate)
    training_iters = request.form['training_iters']
    training_iters = eval(training_iters)
    batch_size= request.form['batch_size']
    batch_size = eval(batch_size)
    display_step = request.form['display_step']
    display_step = eval(display_step)
    
    #network parameters
    #28 * 28 image
    n_input = 784
    n_classes = 10
    dropout = 0.75
    
    x = tf.placeholder(tf.float32,[None, n_input])
    y = tf.placeholder(tf.float32,[None, n_classes])
    keep_prob = tf.placeholder(tf.float32)
    
    def conv2d(x, W, b, strides=1):
        x = tf.nn.conv2d(x,W,strides=[1, strides, strides, 1], padding = 'SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)
    
    def maxpool2d(x, k=2):
        return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')
    
    #create model
    def conv_net(x, weights, biases, dropout):
        #reshape input
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        
        #convolutional layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        #maxpooling
        conv1 = maxpool2d(conv1, k=2)
        
        conv2 = conv2d( conv1, weights['wc2'], biases['bc2'])
        conv2 = maxpool2d(conv2, k=2)
        
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        #apply dropout
        fc1 = tf.nn.dropout(fc1, dropout)
        
        #outpout, class predication
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out
    
    #create weights
    weights = {
            'wc1': tf.Variable(tf.random_normal([5,5,1,32])),
            'wc2': tf.Variable(tf.random_normal([5,5,32,64])),
            'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
            'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }
    
    biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([n_classes]))
      }
    
    #construct model
    pred = conv_net(x, weights, biases, keep_prob)
    
    #define optimizer and loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels = y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    #Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    #initializing the variables
    init = tf.global_variables_initializer()
    
    #style.use('fivethirtyeight')
    
    #Creating Graph
    #fig = plt.figure(edgecolor='c')
    #ax1=fig.add_subplot(1,1,1)
    
    ConvResult=dict()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
                ConvResult[step * batch_size] = acc
            step += 1
        print("Optimization Finished!")
    
        # Calculate accuracy for 256 mnist test images
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                          y: mnist.test.labels[:256],
                                          keep_prob: 1.}))
        test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                          y: mnist.test.labels[:256],
                                          keep_prob: 1.})
    return render_template("ConvNetForm.html", ConvResult=ConvResult, test_accuracy=test_accuracy)
    

@app.route('/MultiPerceptron')
def MultiPerceptron():
    return render_template("MultiPerceptron.html")

@app.route('/MultiPerceptronForm', methods=["POST"])
def MultiPerceptronForm():
    #A Multilayer Perceptron implementation
    # Parameters
    learning_rate = request.form['learning_rate']
    learning_rate=eval(learning_rate)
    training_epochs = request.form['training_epochs']
    training_epochs=eval(training_epochs)
    batch_size = request.form['batch_size']
    batch_size=eval(batch_size)
    display_step = request.form['display_step']
    display_step=eval(display_step)
    
    num_steps_in_one_epoch = int(mnist.train.num_examples/batch_size)
    images_seen_per_epoch = num_steps_in_one_epoch * batch_size
    
    
    # Network Parameters
    n_hidden_1 = 256 # 1st layer number of features
    n_hidden_2 = 256 # 2nd layer number of features
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_classes = 10 # MNIST total classes (0-9 digits)
    
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    
    # Create model
    def multilayer_perceptron(x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    # Construct model
    pred = multilayer_perceptron(x, weights, biases)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    MultiPerceptResult=dict()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
    
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            images_seen_per_epoch = total_batch * batch_size
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                images_seen = epoch * images_seen_per_epoch
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
                # Test model
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
                MultiPerceptResult[images_seen] = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    return render_template('MultiPerceptronForm.html', MultiPerceptResult=MultiPerceptResult)

@app.route('/RNN')
def RNN():
    return render_template("RNN.html")

@app.route('/RNNForm', methods=["POST"])
def RNNForm():
    #A Multilayer Perceptron implementation
    # Parameters
    learning_rate = request.form['learning_rate']
    learning_rate=eval(learning_rate)
    training_iters = request.form['training_iters']
    training_iters=eval(training_iters)
    batch_size = request.form['batch_size']
    batch_size=eval(batch_size)
    display_step = request.form['display_step']
    display_step=eval(display_step)
    
    # Network Parameters
    n_input = 28 # MNIST data input (img shape: 28*28)
    n_steps = 28 # timesteps
    n_hidden = 128 # hidden layer num of features
    n_classes = 10 # MNIST total classes (0-9 digits)
    
    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])
    
    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    
    def RNN(x, weights, biases):
    
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, n_steps, 1)
    
        # Define a lstm cell with tensorflow
        lstm_cell = tf.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    
        # Get lstm cell output
        outputs, states = tf.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    
        # Linear activation, using rnn inner loop last output tf.matmul(outputs[-1], weights['out']) + biases['out']
    
    pred = RNN(x, weights, biases)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    
    RNNresult=dict()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
                RNNresult[step*batch_size]=acc
            step += 1
        print("Optimization Finished!")
    
        # Calculate accuracy for 128 mnist test images
        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        test_label = mnist.test.labels[:test_len]
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
        test_accuracy=sess.run(accuracy, feed_dict={x: test_data, y: test_label})
    return render_template("RNNForm.html", RNNresult=RNNresult, test_accuracy=test_accuracy)


@app.route('/post', methods=['GET', 'POST'])

def post():
    if request.method == 'POST':
        return "You are using POST"
    else:
        return "You are probably using GET"

    #added use_reloader=False to the code on August 18th 2019
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=8080, threaded=True)  

