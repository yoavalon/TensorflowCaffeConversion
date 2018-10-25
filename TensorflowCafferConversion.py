import tensorflow as tf
import numpy as np
import cv2
import os
import random
import sys

sys.path.append('/home/yoav/sources/caffe_bvlc/python')
sys.path.append('/home/yoav/sources/caffe_bvlc/python/caffe')
import caffe

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def transform4(tensor):
    tensor = np.asarray(tensor)
    return np.asarray([tensor[3],tensor[2], tensor[1], tensor[0]])
def transform2(tensor):
    tensor = np.asarray(tensor)
    return np.asarray([tensor[1],tensor[0]])

layers = [("conv_1_0", "triplet/conv1/weights:0"),("relu1_0", "triplet/conv1/biases:0"),("conv_2_0", "triplet/conv2/weights:0"),("relu2_0", "triplet/conv2/biases:0"),("conv_3_0", "triplet/conv3/weights:0"),("relu3_0", "triplet/conv3/biases:0"),("conv_4_0", "triplet/conv4/weights:0"),("relu4_0", "triplet/conv4/biases:0"),("fc_0", "triplet/fc/fc/weights:0")]

net = caffe.Netnet = caffe.Net('deploy32.prototxt',"snapshot_iter_1000.caffemodel",caffe.TEST)

for par in net.params :
    print(par)
    #print(np.asarray(net.blobs[par].data).shape)
    tensor = np.asarray(net.params[par][0].data)
    print(tensor.shape)

with tf.Session() as sess:

    # load the computation graph
    loader = tf.train.import_meta_graph('model12000.ckpt.meta')
    sess.run(tf.global_variables_initializer())
    loader = loader.restore(sess, 'model12000.ckpt')
    
    graph = tf.get_default_graph()
    
    variables = tf.trainable_variables()

    for var in variables :
        print(var.name)
        weights = np.asarray(sess.run(var))
        print(weights.shape)
    
    print("CONVERSION --------------------------------")

    #actual conversion        
    for caf, ten in layers :            

        TensorflowTensor = np.asarray(sess.run(ten))        
        print(TensorflowTensor.shape)

        if(len(TensorflowTensor.shape) == 4) :
            shape = TensorflowTensor.shape
            endshape = transform4(shape)
            TensorflowTensor = TensorflowTensor.reshape((endshape))

        if(len(TensorflowTensor.shape) == 2) :
            shape = TensorflowTensor.shape
            endshape = transform2(shape)        
            TensorflowTensor = TensorflowTensor.reshape((endshape))
       
        print(TensorflowTensor.shape)
        
        #input to caffe weights                
        net.params[caf][0].data[...] = TensorflowTensor
        
    net.save('net.caffemodel') 
    print('done')
