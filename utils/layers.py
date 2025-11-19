import numpy as np
from builtins import range

def affine_forward(x, w, b):
    N = x.shape[0] #sample size
    x_row = x.reshape(N, -1) #flatten input
    out = x_row.dot(w) + b
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    x, w, b = cache
    N = x.shape[0]
    x_row = x.reshape(N, -1)
    
    dw = x_row.T.dot(dout)
    db = np.sum(dout, axis=0)
    dx = dout.dot(w.T).reshape(x.shape)
    
    return dx, dw, db

def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    dx = dout * (cache > 0)
    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps',1e-5)
    momentum = bn_param.get('momentum',0.9)
    
    N,D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    
    out, cache = None, None
    if mode=='train':
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_hat = (x-sample_mean)/np.sqrt(sample_var+eps)
        out = gamma*x_hat + beta
        
        cache = (x, x_hat, gamma, sample_mean, sample_var, eps)
        
        running_mean = momentum*running_mean + (1-momentum)*sample_mean
        running_var = momentum*running_var + (1-momentum)*sample_var
    elif mode=='test':
        x_hat = (x-running_mean)/np.sqrt(running_var+eps)
        out = gamma*x_hat + beta

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    return out, cache

def batchnorm_backward(dout, cache):
    x, x_hat, gamma, mean, var, eps = cache
    N, D = x.shape
    
    dxhat = dout * gamma
    dvar = np.sum(dxhat*(x-mean)*-0.5*(var+eps)**-1.5, axis=0)
    dmean = np.sum(dxhat*-1/np.sqrt(var+eps), axis=0) + dvar*np.sum(-2*(x-mean), axis=0)/N
    dx = dxhat/np.sqrt(var+eps) + dvar*2*(x-mean)/N + dmean/N
    dgamma = np.sum(dout*x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    return dx, dgamma, dbeta

def batchnorm_backward_alt(dout, cache):
    x, x_hat, gamma, mean, var, eps = cache
    N = x.shape[0]
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout*x_hat, axis=0)
    
    dx = (1./N) * gamma * (var+eps)**-0.5 * (N*dout - np.sum(dout, axis=0)
           - (x-mean)*(var+eps)**-1 * np.sum(dout*(x-mean), axis=0))
    return dx, dgamma, dbeta

def layernorm_forward(x, gamma, beta, ln_param):
    eps = ln_param.get('eps',1e-5)
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    x_hat = (x-mean)/np.sqrt(var+eps)
    out = gamma * x_hat + beta
    cache = (x, x_hat, gamma, mean, var, eps)
    return out, cache

def layernorm_backward(dout, cache):
    x, x_hat, gamma, mean, var, eps = cache
    N,D = x.shape
    dxhat = dout*gamma
    dvar = np.sum(dxhat*(x-mean)*-0.5*(var+eps)**-1.5, axis=1, keepdims=True)
    dmean = np.sum(dxhat*-1/np.sqrt(var+eps), axis=1, keepdims=True) + dvar*np.sum(-2*(x-mean), axis=1, keepdims=True)/D
    dx = dxhat/np.sqrt(var+eps) + dvar*2*(x-mean)/D + dmean/D
    dgamma = np.sum(dout*x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])
    mask = None
    if mode=='train':
        mask = (np.random.rand(*x.shape) < p)/p
        out = x*mask
    elif mode=='test':
        out = x
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache

def dropout_backward(dout, cache):
    dropout_param, mask = cache
    mode = dropout_param['mode']
    if mode=='train':
        dx = dout*mask
    else:
        dx = dout
    return dx

def svm_loss(x, y):
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins)/N
    num_pos = np.sum(margins>0, axis=1)
    dx = np.zeros_like(x)
    dx[margins>0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx

def softmax_loss(x, y):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y])/N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

from layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


