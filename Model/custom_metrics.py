# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:44:29 2019

@author: saras
"""
import functools
from keras import metrics

def topk_accuracy(k):
    topk_acc = functools.partial(metrics.top_k_categorical_accuracy, k=5)
    topk_acc.__name__ = 'topk_acc'
    return topk_acc
    
import tensorflow as tf
import keras.backend as K

"""
Tensorflow实现何凯明的Focal Loss, 该损失函数主要用于解决分类问题中的类别不平衡
focal_loss_sigmoid: 二分类loss
focal_loss_softmax: 多分类loss
Reference Paper : Focal Loss for Dense Object Detection
Code: courtesy from:
https://github.com/fudannlp16/focal-loss/blob/master/focal_loss.py 
"""

def focal_loss_sigmoid(labels,logits,alpha=0.25,gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
    L=-labels*(1-alpha)*((1-y_pred)*gamma)*tf.log(y_pred)-\
      (1-labels)*alpha*(y_pred**gamma)*tf.log(1-y_pred)
    return L

def focal_loss_softmax(gamma=2,from_logits=True): ###Multi-class version
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    def focal_loss(labels,logits):
        y_pred=tf.nn.softmax(logits,dim=-1) # [batch_size,num_classes]
        labels=tf.cast(labels,tf.int32)
        y_true=tf.one_hot(labels,depth=y_pred.shape[1])
        L=-y_true*((1-y_pred)**gamma)*tf.log(y_pred)
        L=tf.reduce_sum(L,axis=1)
        return L
    
    def focal_loss_(labels,y_pred):
        labels=tf.cast(labels,tf.int32)
        y_true=tf.one_hot(labels,depth=y_pred.shape[1])
        L=-y_true*((1-y_pred)**gamma)*tf.log(y_pred)
        L=tf.reduce_sum(L,axis=1)
        return L    
    
    if(from_logits==False): return focal_loss_
    return focal_loss

def focal_loss(gamma=2, alpha=0.75):  ###Multi-label version
   def focal_loss_fixed(y_true, y_pred):#with tensorflow
       eps = 1e-12
       y_pred=K.clip(y_pred,eps,1.-eps)#improve the stability of the focal loss and see issues 1 for more information
       pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
       pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
       return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
   return focal_loss_fixed 

#if __name__ == '__main__':
#    logits=tf.random_uniform(shape=[5],minval=-1,maxval=1,dtype=tf.float32)
#    labels=tf.Variable([0,1,0,0,1])
#    loss1=focal_loss_sigmoid(labels=labels,logits=logits)
#
#    logits2=tf.random_uniform(shape=[5,4],minval=-1,maxval=1,dtype=tf.float32)
#    labels2=tf.Variable([1,0,2,3,1])
#    loss2=focal_loss_softmax(labels==labels2,logits=logits2)
#
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        print (sess.run(loss1))
#        print (sess.run(loss2))