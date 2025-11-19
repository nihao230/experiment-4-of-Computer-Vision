from builtins import range
from builtins import object
import os
import numpy as np

from layers import *
from layer_utils import *


class FullyConnectedNet(object):
    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout_keep_ratio=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # 初始化权重和偏置
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params[f'W{i+1}'] = weight_scale * np.random.randn(layer_dims[i], layer_dims[i+1])
            self.params[f'b{i+1}'] = np.zeros(layer_dims[i+1])
            if self.normalization in ['batchnorm', 'layernorm'] and i < self.num_layers-1:
                self.params[f'gamma{i+1}'] = np.ones(layer_dims[i+1])
                self.params[f'beta{i+1}'] = np.zeros(layer_dims[i+1])

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode':'train', 'p':dropout_keep_ratio}
            if seed is not None:
                self.dropout_param['seed'] = seed

        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode':'train'} for _ in range(self.num_layers-1)]
        if self.normalization == 'layernorm':
            self.bn_params = [{} for _ in range(self.num_layers-1)]

        # 类型转换
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        caches = []
        out = X

        # 前向传播
        for i in range(1, self.num_layers):
            W, b = self.params[f'W{i}'], self.params[f'b{i}']
            out, fc_cache = affine_forward(out, W, b)
            if self.normalization == 'batchnorm':
                gamma, beta = self.params[f'gamma{i}'], self.params[f'beta{i}']
                out, bn_cache = batchnorm_forward(out, gamma, beta, self.bn_params[i-1])
                fc_cache = (fc_cache, bn_cache)
            out, relu_cache = relu_forward(out)
            caches.append((fc_cache, relu_cache))
            if self.use_dropout:
                out, do_cache = dropout_forward(out, self.dropout_param)
                caches[-1] += (do_cache,)

        # 最后一层
        W, b = self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}']
        scores, fc_cache = affine_forward(out, W, b)
        caches.append(fc_cache)

        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0
        for i in range(1, self.num_layers+1):
            W = self.params[f'W{i}']
            reg_loss += 0.5 * self.reg * np.sum(W*W)
        loss = data_loss + reg_loss

        # 反向传播
        dout, dW, db = affine_backward(dscores, caches[-1])
        grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = dW + self.reg*self.params[f'W{self.num_layers}'], db

        dout_prev = dout
        for i in reversed(range(1, self.num_layers)):
            cache = caches[i-1]
            if self.use_dropout:
                fc_cache, relu_cache, do_cache = cache
                dout_prev = dropout_backward(dout_prev, do_cache)
            else:
                fc_cache, relu_cache = cache
            dout_prev = relu_backward(dout_prev, relu_cache)
            if self.normalization == 'batchnorm':
                fc_cache, bn_cache = fc_cache
                dout_prev, dgamma, dbeta = batchnorm_backward(dout_prev, bn_cache)
                grads[f'gamma{i}'], grads[f'beta{i}'] = dgamma, dbeta
            dout_prev, dW, db = affine_backward(dout_prev, fc_cache)
            grads[f'W{i}'], grads[f'b{i}'] = dW + self.reg*self.params[f'W{i}'], db

        return loss, grads

    def save(self, fname):
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        np.save(fpath, self.params)
        print(fname, "saved.")

    def load(self, fname):
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        self.params = np.load(fpath, allow_pickle=True).item()
        print(fname, "loaded.")
        return True
