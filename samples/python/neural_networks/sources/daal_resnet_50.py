# file: daal_resnet_50.py
#===============================================================================
# Copyright 2017-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

#
# !  Content:
# !    Python example of neural network training and scoring with ResNet 50 topology
# !*****************************************************************************
from __future__ import division

import os
import sys
import numpy as np

from daal.algorithms.neural_networks import training
from daal.algorithms.neural_networks.layers import (
    batch_normalization, convolution2d, eltwise_sum, relu, maximum_pooling2d,
    fullyconnected, loss, pooling2d, average_pooling2d, split
)
from daal.algorithms.neural_networks.initializers import uniform, xavier

from daal_commons import testClassifier, trainClassifier, batchSize
from blob_dataset import ImageBlobDatasetReader
from service import getUserDatasetPath, selectDatasetPathOrExit


def configureNet():

    topology = training.Topology()

    conv1 = convolution2d.Batch(fptype=np.float32)
    conv1.parameter.nKernels           = 64
    conv1.parameter.kernelSizes        = convolution2d.KernelSizes(7, 7)
    conv1.parameter.strides            = convolution2d.Strides(2, 2)
    conv1.parameter.paddings           = convolution2d.Paddings(3, 3)
    conv1.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    conv1.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    conv1_id = topology.add(conv1)

    bn_conv1 = batch_normalization.Batch(fptype=np.float32)
    bn_conv1_id = topology.add(bn_conv1)

    conv1_relu = relu.Batch(fptype=np.float32)
    conv1_relu_id = topology.add(conv1_relu)

    pool1 = maximum_pooling2d.Batch(4, fptype=np.float32)
    pool1.parameter.kernelSizes = pooling2d.KernelSizes(3, 3)
    pool1.parameter.strides     = pooling2d.Strides(2, 2)
    pool1_id = topology.add(pool1)

    pool1_split1 = split.Batch(2, 2, fptype=np.float32)
    pool1_split1_id = topology.add(pool1_split1)

    res2a_branch1 = convolution2d.Batch(fptype=np.float32)
    res2a_branch1.parameter.nKernels           = 256
    res2a_branch1.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res2a_branch1.parameter.strides            = convolution2d.Strides(1, 1)
    res2a_branch1.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res2a_branch1.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res2a_branch1_id = topology.add(res2a_branch1)

    bn2a_branch1 = batch_normalization.Batch(fptype=np.float32)
    bn2a_branch1_id = topology.add(bn2a_branch1)

    res2a_branch2a = convolution2d.Batch(fptype=np.float32)
    res2a_branch2a.parameter.nKernels           = 64
    res2a_branch2a.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res2a_branch2a.parameter.strides            = convolution2d.Strides(1, 1)
    res2a_branch2a.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res2a_branch2a.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res2a_branch2a_id = topology.add(res2a_branch2a)

    bn2a_branch2a = batch_normalization.Batch(fptype=np.float32)
    bn2a_branch2a_id = topology.add(bn2a_branch2a)

    res2a_branch2a_relu = relu.Batch(fptype=np.float32)
    res2a_branch2a_relu_id = topology.add(res2a_branch2a_relu)

    res2a_branch2b = convolution2d.Batch(fptype=np.float32)
    res2a_branch2b.parameter.nKernels           = 64
    res2a_branch2b.parameter.kernelSizes        = convolution2d.KernelSizes(3, 3)
    res2a_branch2b.parameter.strides            = convolution2d.Strides(1, 1)
    res2a_branch2b.parameter.paddings           = convolution2d.Paddings(1, 1)
    res2a_branch2b.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res2a_branch2b.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res2a_branch2b_id = topology.add(res2a_branch2b)

    bn2a_branch2b = batch_normalization.Batch(fptype=np.float32)
    bn2a_branch2b_id = topology.add(bn2a_branch2b)

    res2a_branch2b_relu = relu.Batch(fptype=np.float32)
    res2a_branch2b_relu_id = topology.add(res2a_branch2b_relu)

    res2a_branch2c = convolution2d.Batch(fptype=np.float32)
    res2a_branch2c.parameter.nKernels           = 256
    res2a_branch2c.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res2a_branch2c.parameter.strides            = convolution2d.Strides(1, 1)
    res2a_branch2c.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res2a_branch2c.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res2a_branch2c_id = topology.add(res2a_branch2c)

    bn2a_branch2c = batch_normalization.Batch(fptype=np.float32)
    bn2a_branch2c_id = topology.add(bn2a_branch2c)

    res2a = eltwise_sum.Batch(fptype=np.float32)
    res2a_id = topology.add(res2a)

    res2a_relu = relu.Batch(fptype=np.float32)
    res2a_relu_id = topology.add(res2a_relu)

    res2a_relu_split2 = split.Batch(2, 2, fptype=np.float32)
    res2a_relu_split2_id = topology.add(res2a_relu_split2)

    res2b_branch2a = convolution2d.Batch(fptype=np.float32)
    res2b_branch2a.parameter.nKernels           = 64
    res2b_branch2a.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res2b_branch2a.parameter.strides            = convolution2d.Strides(1, 1)
    res2b_branch2a.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res2b_branch2a.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res2b_branch2a_id = topology.add(res2b_branch2a)

    bn2b_branch2a = batch_normalization.Batch(fptype=np.float32)
    bn2b_branch2a_id = topology.add(bn2b_branch2a)

    res2b_branch2a_relu = relu.Batch(fptype=np.float32)
    res2b_branch2a_relu_id = topology.add(res2b_branch2a_relu)

    res2b_branch2b = convolution2d.Batch(fptype=np.float32)
    res2b_branch2b.parameter.nKernels           = 64
    res2b_branch2b.parameter.kernelSizes        = convolution2d.KernelSizes(3, 3)
    res2b_branch2b.parameter.strides            = convolution2d.Strides(1, 1)
    res2b_branch2b.parameter.paddings           = convolution2d.Paddings(1, 1)
    res2b_branch2b.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res2b_branch2b.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res2b_branch2b_id = topology.add(res2b_branch2b)

    bn2b_branch2b = batch_normalization.Batch(fptype=np.float32)
    bn2b_branch2b_id = topology.add(bn2b_branch2b)

    res2b_branch2b_relu = relu.Batch(fptype=np.float32)
    res2b_branch2b_relu_id = topology.add(res2b_branch2b_relu)

    res2b_branch2c = convolution2d.Batch(fptype=np.float32)
    res2b_branch2c.parameter.nKernels           = 256
    res2b_branch2c.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res2b_branch2c.parameter.strides            = convolution2d.Strides(1, 1)
    res2b_branch2c.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res2b_branch2c.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res2b_branch2c_id = topology.add(res2b_branch2c)

    bn2b_branch2c = batch_normalization.Batch(fptype=np.float32)
    bn2b_branch2c_id = topology.add(bn2b_branch2c)

    res2b = eltwise_sum.Batch(fptype=np.float32)
    res2b_id = topology.add(res2b)

    res2b_relu = relu.Batch(fptype=np.float32)
    res2b_relu_id = topology.add(res2b_relu)

    res2b_relu_split3 = split.Batch(2, 2, fptype=np.float32)
    res2b_relu_split3_id = topology.add(res2b_relu_split3)

    res2c_branch2a = convolution2d.Batch(fptype=np.float32)
    res2c_branch2a.parameter.nKernels           = 64
    res2c_branch2a.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res2c_branch2a.parameter.strides            = convolution2d.Strides(1, 1)
    res2c_branch2a.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res2c_branch2a.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res2c_branch2a_id = topology.add(res2c_branch2a)

    bn2c_branch2a = batch_normalization.Batch(fptype=np.float32)
    bn2c_branch2a_id = topology.add(bn2c_branch2a)

    res2c_branch2a_relu = relu.Batch(fptype=np.float32)
    res2c_branch2a_relu_id = topology.add(res2c_branch2a_relu)

    res2c_branch2b = convolution2d.Batch(fptype=np.float32)
    res2c_branch2b.parameter.nKernels           = 64
    res2c_branch2b.parameter.kernelSizes        = convolution2d.KernelSizes(3, 3)
    res2c_branch2b.parameter.strides            = convolution2d.Strides(1, 1)
    res2c_branch2b.parameter.paddings           = convolution2d.Paddings(1, 1)
    res2c_branch2b.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res2c_branch2b.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res2c_branch2b_id = topology.add(res2c_branch2b)

    bn2c_branch2b = batch_normalization.Batch(fptype=np.float32)
    bn2c_branch2b_id = topology.add(bn2c_branch2b)

    res2c_branch2b_relu = relu.Batch(fptype=np.float32)
    res2c_branch2b_relu_id = topology.add(res2c_branch2b_relu)

    res2c_branch2c = convolution2d.Batch(fptype=np.float32)
    res2c_branch2c.parameter.nKernels           = 256
    res2c_branch2c.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res2c_branch2c.parameter.strides            = convolution2d.Strides(1, 1)
    res2c_branch2c.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res2c_branch2c.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res2c_branch2c_id = topology.add(res2c_branch2c)

    bn2c_branch2c = batch_normalization.Batch(fptype=np.float32)
    bn2c_branch2c_id = topology.add(bn2c_branch2c)

    res2c = eltwise_sum.Batch(fptype=np.float32)
    res2c_id = topology.add(res2c)

    res2c_relu = relu.Batch(fptype=np.float32)
    res2c_relu_id = topology.add(res2c_relu)

    res2c_relu_split4 = split.Batch(2, 2, fptype=np.float32)
    res2c_relu_split4_id = topology.add(res2c_relu_split4)

    res3a_branch1 = convolution2d.Batch(fptype=np.float32)
    res3a_branch1.parameter.nKernels           = 512
    res3a_branch1.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res3a_branch1.parameter.strides            = convolution2d.Strides(2, 2)
    res3a_branch1.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res3a_branch1.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res3a_branch1_id = topology.add(res3a_branch1)

    bn3a_branch1 = batch_normalization.Batch(fptype=np.float32)
    bn3a_branch1_id = topology.add(bn3a_branch1)

    res3a_branch2a = convolution2d.Batch(fptype=np.float32)
    res3a_branch2a.parameter.nKernels           = 128
    res3a_branch2a.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res3a_branch2a.parameter.strides            = convolution2d.Strides(2, 2)
    res3a_branch2a.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res3a_branch2a.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res3a_branch2a_id = topology.add(res3a_branch2a)

    bn3a_branch2a = batch_normalization.Batch(fptype=np.float32)
    bn3a_branch2a_id = topology.add(bn3a_branch2a)

    res3a_branch2a_relu = relu.Batch(fptype=np.float32)
    res3a_branch2a_relu_id = topology.add(res3a_branch2a_relu)

    res3a_branch2b = convolution2d.Batch(fptype=np.float32)
    res3a_branch2b.parameter.nKernels           = 128
    res3a_branch2b.parameter.kernelSizes        = convolution2d.KernelSizes(3, 3)
    res3a_branch2b.parameter.strides            = convolution2d.Strides(1, 1)
    res3a_branch2b.parameter.paddings           = convolution2d.Paddings(1, 1)
    res3a_branch2b.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res3a_branch2b.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res3a_branch2b_id = topology.add(res3a_branch2b)

    bn3a_branch2b = batch_normalization.Batch(fptype=np.float32)
    bn3a_branch2b_id = topology.add(bn3a_branch2b)

    res3a_branch2b_relu = relu.Batch(fptype=np.float32)
    res3a_branch2b_relu_id = topology.add(res3a_branch2b_relu)

    res3a_branch2c = convolution2d.Batch(fptype=np.float32)
    res3a_branch2c.parameter.nKernels           = 512
    res3a_branch2c.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res3a_branch2c.parameter.strides            = convolution2d.Strides(1, 1)
    res3a_branch2c.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res3a_branch2c.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res3a_branch2c_id = topology.add(res3a_branch2c)

    bn3a_branch2c = batch_normalization.Batch(fptype=np.float32)
    bn3a_branch2c_id = topology.add(bn3a_branch2c)

    res3a = eltwise_sum.Batch(fptype=np.float32)
    res3a_id = topology.add(res3a)

    res3a_relu = relu.Batch(fptype=np.float32)
    res3a_relu_id = topology.add(res3a_relu)

    res3a_relu_split5 = split.Batch(2, 2, fptype=np.float32)
    res3a_relu_split5_id = topology.add(res3a_relu_split5)

    res3b_branch2a = convolution2d.Batch(fptype=np.float32)
    res3b_branch2a.parameter.nKernels           = 128
    res3b_branch2a.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res3b_branch2a.parameter.strides            = convolution2d.Strides(1, 1)
    res3b_branch2a.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res3b_branch2a.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res3b_branch2a_id = topology.add(res3b_branch2a)

    bn3b_branch2a = batch_normalization.Batch(fptype=np.float32)
    bn3b_branch2a_id = topology.add(bn3b_branch2a)

    res3b_branch2a_relu = relu.Batch(fptype=np.float32)
    res3b_branch2a_relu_id = topology.add(res3b_branch2a_relu)

    res3b_branch2b = convolution2d.Batch(fptype=np.float32)
    res3b_branch2b.parameter.nKernels           = 128
    res3b_branch2b.parameter.kernelSizes        = convolution2d.KernelSizes(3, 3)
    res3b_branch2b.parameter.strides            = convolution2d.Strides(1, 1)
    res3b_branch2b.parameter.paddings           = convolution2d.Paddings(1, 1)
    res3b_branch2b.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res3b_branch2b.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res3b_branch2b_id = topology.add(res3b_branch2b)

    bn3b_branch2b = batch_normalization.Batch(fptype=np.float32)
    bn3b_branch2b_id = topology.add(bn3b_branch2b)

    res3b_branch2b_relu = relu.Batch(fptype=np.float32)
    res3b_branch2b_relu_id = topology.add(res3b_branch2b_relu)

    res3b_branch2c = convolution2d.Batch(fptype=np.float32)
    res3b_branch2c.parameter.nKernels           = 512
    res3b_branch2c.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res3b_branch2c.parameter.strides            = convolution2d.Strides(1, 1)
    res3b_branch2c.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res3b_branch2c.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res3b_branch2c_id = topology.add(res3b_branch2c)

    bn3b_branch2c = batch_normalization.Batch(fptype=np.float32)
    bn3b_branch2c_id = topology.add(bn3b_branch2c)

    res3b = eltwise_sum.Batch(fptype=np.float32)
    res3b_id = topology.add(res3b)

    res3b_relu = relu.Batch(fptype=np.float32)
    res3b_relu_id = topology.add(res3b_relu)

    res3b_relu_split6 = split.Batch(2, 2, fptype=np.float32)
    res3b_relu_split6_id = topology.add(res3b_relu_split6)

    res3c_branch2a = convolution2d.Batch(fptype=np.float32)
    res3c_branch2a.parameter.nKernels           = 128
    res3c_branch2a.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res3c_branch2a.parameter.strides            = convolution2d.Strides(1, 1)
    res3c_branch2a.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res3c_branch2a.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res3c_branch2a_id = topology.add(res3c_branch2a)

    bn3c_branch2a = batch_normalization.Batch(fptype=np.float32)
    bn3c_branch2a_id = topology.add(bn3c_branch2a)

    res3c_branch2a_relu = relu.Batch(fptype=np.float32)
    res3c_branch2a_relu_id = topology.add(res3c_branch2a_relu)

    res3c_branch2b = convolution2d.Batch(fptype=np.float32)
    res3c_branch2b.parameter.nKernels           = 128
    res3c_branch2b.parameter.kernelSizes        = convolution2d.KernelSizes(3, 3)
    res3c_branch2b.parameter.strides            = convolution2d.Strides(1, 1)
    res3c_branch2b.parameter.paddings           = convolution2d.Paddings(1, 1)
    res3c_branch2b.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res3c_branch2b.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res3c_branch2b_id = topology.add(res3c_branch2b)

    bn3c_branch2b = batch_normalization.Batch(fptype=np.float32)
    bn3c_branch2b_id = topology.add(bn3c_branch2b)

    res3c_branch2b_relu = relu.Batch(fptype=np.float32)
    res3c_branch2b_relu_id = topology.add(res3c_branch2b_relu)

    res3c_branch2c = convolution2d.Batch(fptype=np.float32)
    res3c_branch2c.parameter.nKernels           = 512
    res3c_branch2c.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res3c_branch2c.parameter.strides            = convolution2d.Strides(1, 1)
    res3c_branch2c.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res3c_branch2c.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res3c_branch2c_id = topology.add(res3c_branch2c)

    bn3c_branch2c = batch_normalization.Batch(fptype=np.float32)
    bn3c_branch2c_id = topology.add(bn3c_branch2c)

    res3c = eltwise_sum.Batch(fptype=np.float32)
    res3c_id = topology.add(res3c)

    res3c_relu = relu.Batch(fptype=np.float32)
    res3c_relu_id = topology.add(res3c_relu)

    res3c_relu_split7 = split.Batch(2, 2, fptype=np.float32)
    res3c_relu_split7_id = topology.add(res3c_relu_split7)

    res3d_branch2a = convolution2d.Batch(fptype=np.float32)
    res3d_branch2a.parameter.nKernels           = 128
    res3d_branch2a.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res3d_branch2a.parameter.strides            = convolution2d.Strides(1, 1)
    res3d_branch2a.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res3d_branch2a.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res3d_branch2a_id = topology.add(res3d_branch2a)

    bn3d_branch2a = batch_normalization.Batch(fptype=np.float32)
    bn3d_branch2a_id = topology.add(bn3d_branch2a)

    res3d_branch2a_relu = relu.Batch(fptype=np.float32)
    res3d_branch2a_relu_id = topology.add(res3d_branch2a_relu)

    res3d_branch2b = convolution2d.Batch(fptype=np.float32)
    res3d_branch2b.parameter.nKernels           = 128
    res3d_branch2b.parameter.kernelSizes        = convolution2d.KernelSizes(3, 3)
    res3d_branch2b.parameter.strides            = convolution2d.Strides(1, 1)
    res3d_branch2b.parameter.paddings           = convolution2d.Paddings(1, 1)
    res3d_branch2b.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res3d_branch2b.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res3d_branch2b_id = topology.add(res3d_branch2b)

    bn3d_branch2b = batch_normalization.Batch(fptype=np.float32)
    bn3d_branch2b_id = topology.add(bn3d_branch2b)

    res3d_branch2b_relu = relu.Batch(fptype=np.float32)
    res3d_branch2b_relu_id = topology.add(res3d_branch2b_relu)

    res3d_branch2c = convolution2d.Batch(fptype=np.float32)
    res3d_branch2c.parameter.nKernels           = 512
    res3d_branch2c.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res3d_branch2c.parameter.strides            = convolution2d.Strides(1, 1)
    res3d_branch2c.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res3d_branch2c.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res3d_branch2c_id = topology.add(res3d_branch2c)

    bn3d_branch2c = batch_normalization.Batch(fptype=np.float32)
    bn3d_branch2c_id = topology.add(bn3d_branch2c)

    res3d = eltwise_sum.Batch(fptype=np.float32)
    res3d_id = topology.add(res3d)

    res3d_relu = relu.Batch(fptype=np.float32)
    res3d_relu_id = topology.add(res3d_relu)

    res3d_relu_split8 = split.Batch(2, 2, fptype=np.float32)
    res3d_relu_split8_id = topology.add(res3d_relu_split8)

    res4a_branch1 = convolution2d.Batch(fptype=np.float32)
    res4a_branch1.parameter.nKernels           = 1024
    res4a_branch1.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res4a_branch1.parameter.strides            = convolution2d.Strides(2, 2)
    res4a_branch1.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4a_branch1.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4a_branch1_id = topology.add(res4a_branch1)

    bn4a_branch1 = batch_normalization.Batch(fptype=np.float32)
    bn4a_branch1_id = topology.add(bn4a_branch1)

    res4a_branch2a = convolution2d.Batch(fptype=np.float32)
    res4a_branch2a.parameter.nKernels           = 256
    res4a_branch2a.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res4a_branch2a.parameter.strides            = convolution2d.Strides(2, 2)
    res4a_branch2a.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4a_branch2a.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4a_branch2a_id = topology.add(res4a_branch2a)

    bn4a_branch2a = batch_normalization.Batch(fptype=np.float32)
    bn4a_branch2a_id = topology.add(bn4a_branch2a)

    res4a_branch2a_relu = relu.Batch(fptype=np.float32)
    res4a_branch2a_relu_id = topology.add(res4a_branch2a_relu)

    res4a_branch2b = convolution2d.Batch(fptype=np.float32)
    res4a_branch2b.parameter.nKernels           = 256
    res4a_branch2b.parameter.kernelSizes        = convolution2d.KernelSizes(3, 3)
    res4a_branch2b.parameter.strides            = convolution2d.Strides(1, 1)
    res4a_branch2b.parameter.paddings           = convolution2d.Paddings(1, 1)
    res4a_branch2b.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4a_branch2b.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4a_branch2b_id = topology.add(res4a_branch2b)

    bn4a_branch2b = batch_normalization.Batch(fptype=np.float32)
    bn4a_branch2b_id = topology.add(bn4a_branch2b)

    res4a_branch2b_relu = relu.Batch(fptype=np.float32)
    res4a_branch2b_relu_id = topology.add(res4a_branch2b_relu)

    res4a_branch2c = convolution2d.Batch(fptype=np.float32)
    res4a_branch2c.parameter.nKernels           = 1024
    res4a_branch2c.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res4a_branch2c.parameter.strides            = convolution2d.Strides(1, 1)
    res4a_branch2c.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4a_branch2c.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4a_branch2c_id = topology.add(res4a_branch2c)

    bn4a_branch2c = batch_normalization.Batch(fptype=np.float32)
    bn4a_branch2c_id = topology.add(bn4a_branch2c)

    res4a = eltwise_sum.Batch(fptype=np.float32)
    res4a_id = topology.add(res4a)

    res4a_relu = relu.Batch(fptype=np.float32)
    res4a_relu_id = topology.add(res4a_relu)

    res4a_relu_split9 = split.Batch(2, 2, fptype=np.float32)
    res4a_relu_split9_id = topology.add(res4a_relu_split9)

    res4b_branch2a = convolution2d.Batch(fptype=np.float32)
    res4b_branch2a.parameter.nKernels           = 256
    res4b_branch2a.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res4b_branch2a.parameter.strides            = convolution2d.Strides(1, 1)
    res4b_branch2a.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4b_branch2a.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4b_branch2a_id = topology.add(res4b_branch2a)

    bn4b_branch2a = batch_normalization.Batch(fptype=np.float32)
    bn4b_branch2a_id = topology.add(bn4b_branch2a)

    res4b_branch2a_relu = relu.Batch(fptype=np.float32)
    res4b_branch2a_relu_id = topology.add(res4b_branch2a_relu)

    res4b_branch2b = convolution2d.Batch(fptype=np.float32)
    res4b_branch2b.parameter.nKernels           = 256
    res4b_branch2b.parameter.kernelSizes        = convolution2d.KernelSizes(3, 3)
    res4b_branch2b.parameter.strides            = convolution2d.Strides(1, 1)
    res4b_branch2b.parameter.paddings           = convolution2d.Paddings(1, 1)
    res4b_branch2b.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4b_branch2b.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4b_branch2b_id = topology.add(res4b_branch2b)

    bn4b_branch2b = batch_normalization.Batch(fptype=np.float32)
    bn4b_branch2b_id = topology.add(bn4b_branch2b)

    res4b_branch2b_relu = relu.Batch(fptype=np.float32)
    res4b_branch2b_relu_id = topology.add(res4b_branch2b_relu)

    res4b_branch2c = convolution2d.Batch(fptype=np.float32)
    res4b_branch2c.parameter.nKernels           = 1024
    res4b_branch2c.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res4b_branch2c.parameter.strides            = convolution2d.Strides(1, 1)
    res4b_branch2c.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4b_branch2c.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4b_branch2c_id = topology.add(res4b_branch2c)

    bn4b_branch2c = batch_normalization.Batch(fptype=np.float32)
    bn4b_branch2c_id = topology.add(bn4b_branch2c)

    res4b = eltwise_sum.Batch(fptype=np.float32)
    res4b_id = topology.add(res4b)

    res4b_relu = relu.Batch(fptype=np.float32)
    res4b_relu_id = topology.add(res4b_relu)

    res4b_relu_split10 = split.Batch(2, 2, fptype=np.float32)
    res4b_relu_split10_id = topology.add(res4b_relu_split10)

    res4c_branch2a = convolution2d.Batch(fptype=np.float32)
    res4c_branch2a.parameter.nKernels           = 256
    res4c_branch2a.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res4c_branch2a.parameter.strides            = convolution2d.Strides(1, 1)
    res4c_branch2a.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4c_branch2a.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4c_branch2a_id = topology.add(res4c_branch2a)

    bn4c_branch2a = batch_normalization.Batch(fptype=np.float32)
    bn4c_branch2a_id = topology.add(bn4c_branch2a)

    res4c_branch2a_relu = relu.Batch(fptype=np.float32)
    res4c_branch2a_relu_id = topology.add(res4c_branch2a_relu)

    res4c_branch2b = convolution2d.Batch(fptype=np.float32)
    res4c_branch2b.parameter.nKernels           = 256
    res4c_branch2b.parameter.kernelSizes        = convolution2d.KernelSizes(3, 3)
    res4c_branch2b.parameter.strides            = convolution2d.Strides(1, 1)
    res4c_branch2b.parameter.paddings           = convolution2d.Paddings(1, 1)
    res4c_branch2b.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4c_branch2b.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4c_branch2b_id = topology.add(res4c_branch2b)

    bn4c_branch2b = batch_normalization.Batch(fptype=np.float32)
    bn4c_branch2b_id = topology.add(bn4c_branch2b)

    res4c_branch2b_relu = relu.Batch(fptype=np.float32)
    res4c_branch2b_relu_id = topology.add(res4c_branch2b_relu)

    res4c_branch2c = convolution2d.Batch(fptype=np.float32)
    res4c_branch2c.parameter.nKernels           = 1024
    res4c_branch2c.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res4c_branch2c.parameter.strides            = convolution2d.Strides(1, 1)
    res4c_branch2c.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4c_branch2c.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4c_branch2c_id = topology.add(res4c_branch2c)

    bn4c_branch2c = batch_normalization.Batch(fptype=np.float32)
    bn4c_branch2c_id = topology.add(bn4c_branch2c)

    res4c = eltwise_sum.Batch(fptype=np.float32)
    res4c_id = topology.add(res4c)

    res4c_relu = relu.Batch(fptype=np.float32)
    res4c_relu_id = topology.add(res4c_relu)

    res4c_relu_split11 = split.Batch(2, 2, fptype=np.float32)
    res4c_relu_split11_id = topology.add(res4c_relu_split11)

    res4d_branch2a = convolution2d.Batch(fptype=np.float32)
    res4d_branch2a.parameter.nKernels           = 256
    res4d_branch2a.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res4d_branch2a.parameter.strides            = convolution2d.Strides(1, 1)
    res4d_branch2a.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4d_branch2a.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4d_branch2a_id = topology.add(res4d_branch2a)

    bn4d_branch2a = batch_normalization.Batch(fptype=np.float32)
    bn4d_branch2a_id = topology.add(bn4d_branch2a)

    res4d_branch2a_relu = relu.Batch(fptype=np.float32)
    res4d_branch2a_relu_id = topology.add(res4d_branch2a_relu)

    res4d_branch2b = convolution2d.Batch(fptype=np.float32)
    res4d_branch2b.parameter.nKernels           = 256
    res4d_branch2b.parameter.kernelSizes        = convolution2d.KernelSizes(3, 3)
    res4d_branch2b.parameter.strides            = convolution2d.Strides(1, 1)
    res4d_branch2b.parameter.paddings           = convolution2d.Paddings(1, 1)
    res4d_branch2b.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4d_branch2b.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4d_branch2b_id = topology.add(res4d_branch2b)

    bn4d_branch2b = batch_normalization.Batch(fptype=np.float32)
    bn4d_branch2b_id = topology.add(bn4d_branch2b)

    res4d_branch2b_relu = relu.Batch(fptype=np.float32)
    res4d_branch2b_relu_id = topology.add(res4d_branch2b_relu)

    res4d_branch2c = convolution2d.Batch(fptype=np.float32)
    res4d_branch2c.parameter.nKernels           = 1024
    res4d_branch2c.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res4d_branch2c.parameter.strides            = convolution2d.Strides(1, 1)
    res4d_branch2c.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4d_branch2c.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4d_branch2c_id = topology.add(res4d_branch2c)

    bn4d_branch2c = batch_normalization.Batch(fptype=np.float32)
    bn4d_branch2c_id = topology.add(bn4d_branch2c)

    res4d = eltwise_sum.Batch(fptype=np.float32)
    res4d_id = topology.add(res4d)

    res4d_relu = relu.Batch(fptype=np.float32)
    res4d_relu_id = topology.add(res4d_relu)

    res4d_relu_split12 = split.Batch(2, 2, fptype=np.float32)
    res4d_relu_split12_id = topology.add(res4d_relu_split12)

    res4e_branch2a = convolution2d.Batch(fptype=np.float32)
    res4e_branch2a.parameter.nKernels           = 256
    res4e_branch2a.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res4e_branch2a.parameter.strides            = convolution2d.Strides(1, 1)
    res4e_branch2a.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4e_branch2a.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4e_branch2a_id = topology.add(res4e_branch2a)

    bn4e_branch2a = batch_normalization.Batch(fptype=np.float32)
    bn4e_branch2a_id = topology.add(bn4e_branch2a)

    res4e_branch2a_relu = relu.Batch(fptype=np.float32)
    res4e_branch2a_relu_id = topology.add(res4e_branch2a_relu)

    res4e_branch2b = convolution2d.Batch(fptype=np.float32)
    res4e_branch2b.parameter.nKernels           = 256
    res4e_branch2b.parameter.kernelSizes        = convolution2d.KernelSizes(3, 3)
    res4e_branch2b.parameter.strides            = convolution2d.Strides(1, 1)
    res4e_branch2b.parameter.paddings           = convolution2d.Paddings(1, 1)
    res4e_branch2b.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4e_branch2b.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4e_branch2b_id = topology.add(res4e_branch2b)

    bn4e_branch2b = batch_normalization.Batch(fptype=np.float32)
    bn4e_branch2b_id = topology.add(bn4e_branch2b)

    res4e_branch2b_relu = relu.Batch(fptype=np.float32)
    res4e_branch2b_relu_id = topology.add(res4e_branch2b_relu)

    res4e_branch2c = convolution2d.Batch(fptype=np.float32)
    res4e_branch2c.parameter.nKernels           = 1024
    res4e_branch2c.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res4e_branch2c.parameter.strides            = convolution2d.Strides(1, 1)
    res4e_branch2c.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4e_branch2c.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4e_branch2c_id = topology.add(res4e_branch2c)

    bn4e_branch2c = batch_normalization.Batch(fptype=np.float32)
    bn4e_branch2c_id = topology.add(bn4e_branch2c)

    res4e = eltwise_sum.Batch(fptype=np.float32)
    res4e_id = topology.add(res4e)

    res4e_relu = relu.Batch(fptype=np.float32)
    res4e_relu_id = topology.add(res4e_relu)

    res4e_relu_split13 = split.Batch(2, 2, fptype=np.float32)
    res4e_relu_split13_id = topology.add(res4e_relu_split13)

    res4f_branch2a = convolution2d.Batch(fptype=np.float32)
    res4f_branch2a.parameter.nKernels           = 256
    res4f_branch2a.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res4f_branch2a.parameter.strides            = convolution2d.Strides(1, 1)
    res4f_branch2a.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4f_branch2a.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4f_branch2a_id = topology.add(res4f_branch2a)

    bn4f_branch2a = batch_normalization.Batch(fptype=np.float32)
    bn4f_branch2a_id = topology.add(bn4f_branch2a)

    res4f_branch2a_relu = relu.Batch(fptype=np.float32)
    res4f_branch2a_relu_id = topology.add(res4f_branch2a_relu)

    res4f_branch2b = convolution2d.Batch(fptype=np.float32)
    res4f_branch2b.parameter.nKernels           = 256
    res4f_branch2b.parameter.kernelSizes        = convolution2d.KernelSizes(3, 3)
    res4f_branch2b.parameter.strides            = convolution2d.Strides(1, 1)
    res4f_branch2b.parameter.paddings           = convolution2d.Paddings(1, 1)
    res4f_branch2b.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4f_branch2b.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4f_branch2b_id = topology.add(res4f_branch2b)

    bn4f_branch2b = batch_normalization.Batch(fptype=np.float32)
    bn4f_branch2b_id = topology.add(bn4f_branch2b)

    res4f_branch2b_relu = relu.Batch(fptype=np.float32)
    res4f_branch2b_relu_id = topology.add(res4f_branch2b_relu)

    res4f_branch2c = convolution2d.Batch(fptype=np.float32)
    res4f_branch2c.parameter.nKernels           = 1024
    res4f_branch2c.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res4f_branch2c.parameter.strides            = convolution2d.Strides(1, 1)
    res4f_branch2c.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res4f_branch2c.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res4f_branch2c_id = topology.add(res4f_branch2c)

    bn4f_branch2c = batch_normalization.Batch(fptype=np.float32)
    bn4f_branch2c_id = topology.add(bn4f_branch2c)

    res4f = eltwise_sum.Batch(fptype=np.float32)
    res4f_id = topology.add(res4f)

    res4f_relu = relu.Batch(fptype=np.float32)
    res4f_relu_id = topology.add(res4f_relu)

    res4f_relu_split14 = split.Batch(2, 2, fptype=np.float32)
    res4f_relu_split14_id = topology.add(res4f_relu_split14)

    res5a_branch1 = convolution2d.Batch(fptype=np.float32)
    res5a_branch1.parameter.nKernels           = 2048
    res5a_branch1.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res5a_branch1.parameter.strides            = convolution2d.Strides(2, 2)
    res5a_branch1.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res5a_branch1.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res5a_branch1_id = topology.add(res5a_branch1)

    bn5a_branch1 = batch_normalization.Batch(fptype=np.float32)
    bn5a_branch1_id = topology.add(bn5a_branch1)

    res5a_branch2a = convolution2d.Batch(fptype=np.float32)
    res5a_branch2a.parameter.nKernels           = 512
    res5a_branch2a.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res5a_branch2a.parameter.strides            = convolution2d.Strides(2, 2)
    res5a_branch2a.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res5a_branch2a.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res5a_branch2a_id = topology.add(res5a_branch2a)

    bn5a_branch2a = batch_normalization.Batch(fptype=np.float32)
    bn5a_branch2a_id = topology.add(bn5a_branch2a)

    res5a_branch2a_relu = relu.Batch(fptype=np.float32)
    res5a_branch2a_relu_id = topology.add(res5a_branch2a_relu)

    res5a_branch2b = convolution2d.Batch(fptype=np.float32)
    res5a_branch2b.parameter.nKernels           = 512
    res5a_branch2b.parameter.kernelSizes        = convolution2d.KernelSizes(3, 3)
    res5a_branch2b.parameter.strides            = convolution2d.Strides(1, 1)
    res5a_branch2b.parameter.paddings           = convolution2d.Paddings(1, 1)
    res5a_branch2b.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res5a_branch2b.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res5a_branch2b_id = topology.add(res5a_branch2b)

    bn5a_branch2b = batch_normalization.Batch(fptype=np.float32)
    bn5a_branch2b_id = topology.add(bn5a_branch2b)

    res5a_branch2b_relu = relu.Batch(fptype=np.float32)
    res5a_branch2b_relu_id = topology.add(res5a_branch2b_relu)

    res5a_branch2c = convolution2d.Batch(fptype=np.float32)
    res5a_branch2c.parameter.nKernels           = 2048
    res5a_branch2c.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res5a_branch2c.parameter.strides            = convolution2d.Strides(1, 1)
    res5a_branch2c.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res5a_branch2c.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res5a_branch2c_id = topology.add(res5a_branch2c)

    bn5a_branch2c = batch_normalization.Batch(fptype=np.float32)
    bn5a_branch2c_id = topology.add(bn5a_branch2c)

    res5a = eltwise_sum.Batch(fptype=np.float32)
    res5a_id = topology.add(res5a)

    res5a_relu = relu.Batch(fptype=np.float32)
    res5a_relu_id = topology.add(res5a_relu)

    res5a_relu_split15 = split.Batch(2, 2, fptype=np.float32)
    res5a_relu_split15_id = topology.add(res5a_relu_split15)

    res5b_branch2a = convolution2d.Batch(fptype=np.float32)
    res5b_branch2a.parameter.nKernels           = 512
    res5b_branch2a.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res5b_branch2a.parameter.strides            = convolution2d.Strides(1, 1)
    res5b_branch2a.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res5b_branch2a.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res5b_branch2a_id = topology.add(res5b_branch2a)

    bn5b_branch2a = batch_normalization.Batch(fptype=np.float32)
    bn5b_branch2a_id = topology.add(bn5b_branch2a)

    res5b_branch2a_relu = relu.Batch(fptype=np.float32)
    res5b_branch2a_relu_id = topology.add(res5b_branch2a_relu)

    res5b_branch2b = convolution2d.Batch(fptype=np.float32)
    res5b_branch2b.parameter.nKernels           = 512
    res5b_branch2b.parameter.kernelSizes        = convolution2d.KernelSizes(3, 3)
    res5b_branch2b.parameter.strides            = convolution2d.Strides(1, 1)
    res5b_branch2b.parameter.paddings           = convolution2d.Paddings(1, 1)
    res5b_branch2b.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res5b_branch2b.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res5b_branch2b_id = topology.add(res5b_branch2b)

    bn5b_branch2b = batch_normalization.Batch(fptype=np.float32)
    bn5b_branch2b_id = topology.add(bn5b_branch2b)

    res5b_branch2b_relu = relu.Batch(fptype=np.float32)
    res5b_branch2b_relu_id = topology.add(res5b_branch2b_relu)

    res5b_branch2c = convolution2d.Batch(fptype=np.float32)
    res5b_branch2c.parameter.nKernels           = 2048
    res5b_branch2c.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res5b_branch2c.parameter.strides            = convolution2d.Strides(1, 1)
    res5b_branch2c.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res5b_branch2c.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res5b_branch2c_id = topology.add(res5b_branch2c)

    bn5b_branch2c = batch_normalization.Batch(fptype=np.float32)
    bn5b_branch2c_id = topology.add(bn5b_branch2c)

    res5b = eltwise_sum.Batch(fptype=np.float32)
    res5b_id = topology.add(res5b)

    res5b_relu = relu.Batch(fptype=np.float32)
    res5b_relu_id = topology.add(res5b_relu)

    res5b_relu_split16 = split.Batch(2, 2, fptype=np.float32)
    res5b_relu_split16_id = topology.add(res5b_relu_split16)

    res5c_branch2a = convolution2d.Batch(fptype=np.float32)
    res5c_branch2a.parameter.nKernels           = 512
    res5c_branch2a.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res5c_branch2a.parameter.strides            = convolution2d.Strides(1, 1)
    res5c_branch2a.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res5c_branch2a.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res5c_branch2a_id = topology.add(res5c_branch2a)

    bn5c_branch2a = batch_normalization.Batch(fptype=np.float32)
    bn5c_branch2a_id = topology.add(bn5c_branch2a)

    res5c_branch2a_relu = relu.Batch(fptype=np.float32)
    res5c_branch2a_relu_id = topology.add(res5c_branch2a_relu)

    res5c_branch2b = convolution2d.Batch(fptype=np.float32)
    res5c_branch2b.parameter.nKernels           = 512
    res5c_branch2b.parameter.kernelSizes        = convolution2d.KernelSizes(3, 3)
    res5c_branch2b.parameter.strides            = convolution2d.Strides(1, 1)
    res5c_branch2b.parameter.paddings           = convolution2d.Paddings(1, 1)
    res5c_branch2b.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res5c_branch2b.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res5c_branch2b_id = topology.add(res5c_branch2b)

    bn5c_branch2b = batch_normalization.Batch(fptype=np.float32)
    bn5c_branch2b_id = topology.add(bn5c_branch2b)

    res5c_branch2b_relu = relu.Batch(fptype=np.float32)
    res5c_branch2b_relu_id = topology.add(res5c_branch2b_relu)

    res5c_branch2c = convolution2d.Batch(fptype=np.float32)
    res5c_branch2c.parameter.nKernels           = 2048
    res5c_branch2c.parameter.kernelSizes        = convolution2d.KernelSizes(1, 1)
    res5c_branch2c.parameter.strides            = convolution2d.Strides(1, 1)
    res5c_branch2c.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    res5c_branch2c.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    res5c_branch2c_id = topology.add(res5c_branch2c)

    bn5c_branch2c = batch_normalization.Batch(fptype=np.float32)
    bn5c_branch2c_id = topology.add(bn5c_branch2c)

    res5c = eltwise_sum.Batch(fptype=np.float32)
    res5c_id = topology.add(res5c)

    res5c_relu = relu.Batch(fptype=np.float32)
    res5c_relu_id = topology.add(res5c_relu)

    pool5 = average_pooling2d.Batch(4, fptype=np.float32)
    pool5.parameter.kernelSizes = pooling2d.KernelSizes(7, 7)
    pool5.parameter.strides     = pooling2d.Strides(1, 1)
    pool5_id = topology.add(pool5)

    fc1000 = fullyconnected.Batch(1000, fptype=np.float32)
    fc1000.parameter.weightsInitializer = xavier.Batch(fptype=np.float32)
    fc1000.parameter.biasesInitializer  = uniform.Batch(0, 0, fptype=np.float32)
    fc1000_id = topology.add(fc1000)

    loss_layer = loss.softmax_cross.Batch(fptype=np.float32)
    loss_layer_id = topology.add(loss_layer)

    topology.get( conv1_id               ).addNext( bn_conv1_id            )
    topology.get( bn_conv1_id            ).addNext( conv1_relu_id          )
    topology.get( conv1_relu_id          ).addNext( pool1_id               )
    topology.get( pool1_id               ).addNext( pool1_split1_id        )
    topology.get( pool1_split1_id        ).addNext( res2a_branch2a_id      )
    topology.get( pool1_split1_id        ).addNext( res2a_branch1_id       )
    topology.get( res2a_branch1_id       ).addNext( bn2a_branch1_id        )
    topology.get( bn2a_branch1_id        ).addNext( res2a_id               )
    topology.get( res2a_branch2a_id      ).addNext( bn2a_branch2a_id       )
    topology.get( bn2a_branch2a_id       ).addNext( res2a_branch2a_relu_id )
    topology.get( res2a_branch2a_relu_id ).addNext( res2a_branch2b_id      )
    topology.get( res2a_branch2b_id      ).addNext( bn2a_branch2b_id       )
    topology.get( bn2a_branch2b_id       ).addNext( res2a_branch2b_relu_id )
    topology.get( res2a_branch2b_relu_id ).addNext( res2a_branch2c_id      )
    topology.get( res2a_branch2c_id      ).addNext( bn2a_branch2c_id       )
    topology.get( bn2a_branch2c_id       ).addNext( res2a_id               )
    topology.get( res2a_id               ).addNext( res2a_relu_id          )
    topology.get( res2a_relu_id          ).addNext( res2a_relu_split2_id   )
    topology.get( res2a_relu_split2_id   ).addNext( res2b_branch2a_id      )
    topology.get( res2a_relu_split2_id   ).addNext( res2b_id               )
    topology.get( res2b_branch2a_id      ).addNext( bn2b_branch2a_id       )
    topology.get( bn2b_branch2a_id       ).addNext( res2b_branch2a_relu_id )
    topology.get( res2b_branch2a_relu_id ).addNext( res2b_branch2b_id      )
    topology.get( res2b_branch2b_id      ).addNext( bn2b_branch2b_id       )
    topology.get( bn2b_branch2b_id       ).addNext( res2b_branch2b_relu_id )
    topology.get( res2b_branch2b_relu_id ).addNext( res2b_branch2c_id      )
    topology.get( res2b_branch2c_id      ).addNext( bn2b_branch2c_id       )
    topology.get( bn2b_branch2c_id       ).addNext( res2b_id               )
    topology.get( res2b_id               ).addNext( res2b_relu_id          )
    topology.get( res2b_relu_id          ).addNext( res2b_relu_split3_id   )
    topology.get( res2b_relu_split3_id   ).addNext( res2c_id               )
    topology.get( res2b_relu_split3_id   ).addNext( res2c_branch2a_id      )
    topology.get( res2c_branch2a_id      ).addNext( bn2c_branch2a_id       )
    topology.get( bn2c_branch2a_id       ).addNext( res2c_branch2a_relu_id )
    topology.get( res2c_branch2a_relu_id ).addNext( res2c_branch2b_id      )
    topology.get( res2c_branch2b_id      ).addNext( bn2c_branch2b_id       )
    topology.get( bn2c_branch2b_id       ).addNext( res2c_branch2b_relu_id )
    topology.get( res2c_branch2b_relu_id ).addNext( res2c_branch2c_id      )
    topology.get( res2c_branch2c_id      ).addNext( bn2c_branch2c_id       )
    topology.get( bn2c_branch2c_id       ).addNext( res2c_id               )
    topology.get( res2c_id               ).addNext( res2c_relu_id          )
    topology.get( res2c_relu_id          ).addNext( res2c_relu_split4_id   )
    topology.get( res2c_relu_split4_id   ).addNext( res3a_branch1_id       )
    topology.get( res2c_relu_split4_id   ).addNext( res3a_branch2a_id      )
    topology.get( res3a_branch1_id       ).addNext( bn3a_branch1_id        )
    topology.get( bn3a_branch1_id        ).addNext( res3a_id               )
    topology.get( res3a_branch2a_id      ).addNext( bn3a_branch2a_id       )
    topology.get( bn3a_branch2a_id       ).addNext( res3a_branch2a_relu_id )
    topology.get( res3a_branch2a_relu_id ).addNext( res3a_branch2b_id      )
    topology.get( res3a_branch2b_id      ).addNext( bn3a_branch2b_id       )
    topology.get( bn3a_branch2b_id       ).addNext( res3a_branch2b_relu_id )
    topology.get( res3a_branch2b_relu_id ).addNext( res3a_branch2c_id      )
    topology.get( res3a_branch2c_id      ).addNext( bn3a_branch2c_id       )
    topology.get( bn3a_branch2c_id       ).addNext( res3a_id               )
    topology.get( res3a_id               ).addNext( res3a_relu_id          )
    topology.get( res3a_relu_id          ).addNext( res3a_relu_split5_id   )
    topology.get( res3a_relu_split5_id   ).addNext( res3b_id               )
    topology.get( res3a_relu_split5_id   ).addNext( res3b_branch2a_id      )
    topology.get( res3b_branch2a_id      ).addNext( bn3b_branch2a_id       )
    topology.get( bn3b_branch2a_id       ).addNext( res3b_branch2a_relu_id )
    topology.get( res3b_branch2a_relu_id ).addNext( res3b_branch2b_id      )
    topology.get( res3b_branch2b_id      ).addNext( bn3b_branch2b_id       )
    topology.get( bn3b_branch2b_id       ).addNext( res3b_branch2b_relu_id )
    topology.get( res3b_branch2b_relu_id ).addNext( res3b_branch2c_id      )
    topology.get( res3b_branch2c_id      ).addNext( bn3b_branch2c_id       )
    topology.get( bn3b_branch2c_id       ).addNext( res3b_id               )
    topology.get( res3b_id               ).addNext( res3b_relu_id          )
    topology.get( res3b_relu_id          ).addNext( res3b_relu_split6_id   )
    topology.get( res3b_relu_split6_id   ).addNext( res3c_id               )
    topology.get( res3b_relu_split6_id   ).addNext( res3c_branch2a_id      )
    topology.get( res3c_branch2a_id      ).addNext( bn3c_branch2a_id       )
    topology.get( bn3c_branch2a_id       ).addNext( res3c_branch2a_relu_id )
    topology.get( res3c_branch2a_relu_id ).addNext( res3c_branch2b_id      )
    topology.get( res3c_branch2b_id      ).addNext( bn3c_branch2b_id       )
    topology.get( bn3c_branch2b_id       ).addNext( res3c_branch2b_relu_id )
    topology.get( res3c_branch2b_relu_id ).addNext( res3c_branch2c_id      )
    topology.get( res3c_branch2c_id      ).addNext( bn3c_branch2c_id       )
    topology.get( bn3c_branch2c_id       ).addNext( res3c_id               )
    topology.get( res3c_id               ).addNext( res3c_relu_id          )
    topology.get( res3c_relu_id          ).addNext( res3c_relu_split7_id   )
    topology.get( res3c_relu_split7_id   ).addNext( res3d_branch2a_id      )
    topology.get( res3c_relu_split7_id   ).addNext( res3d_id               )
    topology.get( res3d_branch2a_id      ).addNext( bn3d_branch2a_id       )
    topology.get( bn3d_branch2a_id       ).addNext( res3d_branch2a_relu_id )
    topology.get( res3d_branch2a_relu_id ).addNext( res3d_branch2b_id      )
    topology.get( res3d_branch2b_id      ).addNext( bn3d_branch2b_id       )
    topology.get( bn3d_branch2b_id       ).addNext( res3d_branch2b_relu_id )
    topology.get( res3d_branch2b_relu_id ).addNext( res3d_branch2c_id      )
    topology.get( res3d_branch2c_id      ).addNext( bn3d_branch2c_id       )
    topology.get( bn3d_branch2c_id       ).addNext( res3d_id               )
    topology.get( res3d_id               ).addNext( res3d_relu_id          )
    topology.get( res3d_relu_id          ).addNext( res3d_relu_split8_id   )
    topology.get( res3d_relu_split8_id   ).addNext( res4a_branch1_id       )
    topology.get( res3d_relu_split8_id   ).addNext( res4a_branch2a_id      )
    topology.get( res4a_branch1_id       ).addNext( bn4a_branch1_id        )
    topology.get( bn4a_branch1_id        ).addNext( res4a_id               )
    topology.get( res4a_branch2a_id      ).addNext( bn4a_branch2a_id       )
    topology.get( bn4a_branch2a_id       ).addNext( res4a_branch2a_relu_id )
    topology.get( res4a_branch2a_relu_id ).addNext( res4a_branch2b_id      )
    topology.get( res4a_branch2b_id      ).addNext( bn4a_branch2b_id       )
    topology.get( bn4a_branch2b_id       ).addNext( res4a_branch2b_relu_id )
    topology.get( res4a_branch2b_relu_id ).addNext( res4a_branch2c_id      )
    topology.get( res4a_branch2c_id      ).addNext( bn4a_branch2c_id       )
    topology.get( bn4a_branch2c_id       ).addNext( res4a_id               )
    topology.get( res4a_id               ).addNext( res4a_relu_id          )
    topology.get( res4a_relu_id          ).addNext( res4a_relu_split9_id   )
    topology.get( res4a_relu_split9_id   ).addNext( res4b_id               )
    topology.get( res4a_relu_split9_id   ).addNext( res4b_branch2a_id      )
    topology.get( res4b_branch2a_id      ).addNext( bn4b_branch2a_id       )
    topology.get( bn4b_branch2a_id       ).addNext( res4b_branch2a_relu_id )
    topology.get( res4b_branch2a_relu_id ).addNext( res4b_branch2b_id      )
    topology.get( res4b_branch2b_id      ).addNext( bn4b_branch2b_id       )
    topology.get( bn4b_branch2b_id       ).addNext( res4b_branch2b_relu_id )
    topology.get( res4b_branch2b_relu_id ).addNext( res4b_branch2c_id      )
    topology.get( res4b_branch2c_id      ).addNext( bn4b_branch2c_id       )
    topology.get( bn4b_branch2c_id       ).addNext( res4b_id               )
    topology.get( res4b_id               ).addNext( res4b_relu_id          )
    topology.get( res4b_relu_id          ).addNext( res4b_relu_split10_id  )
    topology.get( res4b_relu_split10_id  ).addNext( res4c_branch2a_id      )
    topology.get( res4b_relu_split10_id  ).addNext( res4c_id               )
    topology.get( res4c_branch2a_id      ).addNext( bn4c_branch2a_id       )
    topology.get( bn4c_branch2a_id       ).addNext( res4c_branch2a_relu_id )
    topology.get( res4c_branch2a_relu_id ).addNext( res4c_branch2b_id      )
    topology.get( res4c_branch2b_id      ).addNext( bn4c_branch2b_id       )
    topology.get( bn4c_branch2b_id       ).addNext( res4c_branch2b_relu_id )
    topology.get( res4c_branch2b_relu_id ).addNext( res4c_branch2c_id      )
    topology.get( res4c_branch2c_id      ).addNext( bn4c_branch2c_id       )
    topology.get( bn4c_branch2c_id       ).addNext( res4c_id               )
    topology.get( res4c_id               ).addNext( res4c_relu_id          )
    topology.get( res4c_relu_id          ).addNext( res4c_relu_split11_id  )
    topology.get( res4c_relu_split11_id  ).addNext( res4d_id               )
    topology.get( res4c_relu_split11_id  ).addNext( res4d_branch2a_id      )
    topology.get( res4d_branch2a_id      ).addNext( bn4d_branch2a_id       )
    topology.get( bn4d_branch2a_id       ).addNext( res4d_branch2a_relu_id )
    topology.get( res4d_branch2a_relu_id ).addNext( res4d_branch2b_id      )
    topology.get( res4d_branch2b_id      ).addNext( bn4d_branch2b_id       )
    topology.get( bn4d_branch2b_id       ).addNext( res4d_branch2b_relu_id )
    topology.get( res4d_branch2b_relu_id ).addNext( res4d_branch2c_id      )
    topology.get( res4d_branch2c_id      ).addNext( bn4d_branch2c_id       )
    topology.get( bn4d_branch2c_id       ).addNext( res4d_id               )
    topology.get( res4d_id               ).addNext( res4d_relu_id          )
    topology.get( res4d_relu_id          ).addNext( res4d_relu_split12_id  )
    topology.get( res4d_relu_split12_id  ).addNext( res4e_branch2a_id      )
    topology.get( res4d_relu_split12_id  ).addNext( res4e_id               )
    topology.get( res4e_branch2a_id      ).addNext( bn4e_branch2a_id       )
    topology.get( bn4e_branch2a_id       ).addNext( res4e_branch2a_relu_id )
    topology.get( res4e_branch2a_relu_id ).addNext( res4e_branch2b_id      )
    topology.get( res4e_branch2b_id      ).addNext( bn4e_branch2b_id       )
    topology.get( bn4e_branch2b_id       ).addNext( res4e_branch2b_relu_id )
    topology.get( res4e_branch2b_relu_id ).addNext( res4e_branch2c_id      )
    topology.get( res4e_branch2c_id      ).addNext( bn4e_branch2c_id       )
    topology.get( bn4e_branch2c_id       ).addNext( res4e_id               )
    topology.get( res4e_id               ).addNext( res4e_relu_id          )
    topology.get( res4e_relu_id          ).addNext( res4e_relu_split13_id  )
    topology.get( res4e_relu_split13_id  ).addNext( res4f_id               )
    topology.get( res4e_relu_split13_id  ).addNext( res4f_branch2a_id      )
    topology.get( res4f_branch2a_id      ).addNext( bn4f_branch2a_id       )
    topology.get( bn4f_branch2a_id       ).addNext( res4f_branch2a_relu_id )
    topology.get( res4f_branch2a_relu_id ).addNext( res4f_branch2b_id      )
    topology.get( res4f_branch2b_id      ).addNext( bn4f_branch2b_id       )
    topology.get( bn4f_branch2b_id       ).addNext( res4f_branch2b_relu_id )
    topology.get( res4f_branch2b_relu_id ).addNext( res4f_branch2c_id      )
    topology.get( res4f_branch2c_id      ).addNext( bn4f_branch2c_id       )
    topology.get( bn4f_branch2c_id       ).addNext( res4f_id               )
    topology.get( res4f_id               ).addNext( res4f_relu_id          )
    topology.get( res4f_relu_id          ).addNext( res4f_relu_split14_id  )
    topology.get( res4f_relu_split14_id  ).addNext( res5a_branch1_id       )
    topology.get( res4f_relu_split14_id  ).addNext( res5a_branch2a_id      )
    topology.get( res5a_branch1_id       ).addNext( bn5a_branch1_id        )
    topology.get( bn5a_branch1_id        ).addNext( res5a_id               )
    topology.get( res5a_branch2a_id      ).addNext( bn5a_branch2a_id       )
    topology.get( bn5a_branch2a_id       ).addNext( res5a_branch2a_relu_id )
    topology.get( res5a_branch2a_relu_id ).addNext( res5a_branch2b_id      )
    topology.get( res5a_branch2b_id      ).addNext( bn5a_branch2b_id       )
    topology.get( bn5a_branch2b_id       ).addNext( res5a_branch2b_relu_id )
    topology.get( res5a_branch2b_relu_id ).addNext( res5a_branch2c_id      )
    topology.get( res5a_branch2c_id      ).addNext( bn5a_branch2c_id       )
    topology.get( bn5a_branch2c_id       ).addNext( res5a_id               )
    topology.get( res5a_id               ).addNext( res5a_relu_id          )
    topology.get( res5a_relu_id          ).addNext( res5a_relu_split15_id  )
    topology.get( res5a_relu_split15_id  ).addNext( res5b_branch2a_id      )
    topology.get( res5a_relu_split15_id  ).addNext( res5b_id               )
    topology.get( res5b_branch2a_id      ).addNext( bn5b_branch2a_id       )
    topology.get( bn5b_branch2a_id       ).addNext( res5b_branch2a_relu_id )
    topology.get( res5b_branch2a_relu_id ).addNext( res5b_branch2b_id      )
    topology.get( res5b_branch2b_id      ).addNext( bn5b_branch2b_id       )
    topology.get( bn5b_branch2b_id       ).addNext( res5b_branch2b_relu_id )
    topology.get( res5b_branch2b_relu_id ).addNext( res5b_branch2c_id      )
    topology.get( res5b_branch2c_id      ).addNext( bn5b_branch2c_id       )
    topology.get( bn5b_branch2c_id       ).addNext( res5b_id               )
    topology.get( res5b_id               ).addNext( res5b_relu_id          )
    topology.get( res5b_relu_id          ).addNext( res5b_relu_split16_id  )
    topology.get( res5b_relu_split16_id  ).addNext( res5c_branch2a_id      )
    topology.get( res5b_relu_split16_id  ).addNext( res5c_id               )
    topology.get( res5c_branch2a_id      ).addNext( bn5c_branch2a_id       )
    topology.get( bn5c_branch2a_id       ).addNext( res5c_branch2a_relu_id )
    topology.get( res5c_branch2a_relu_id ).addNext( res5c_branch2b_id      )
    topology.get( res5c_branch2b_id      ).addNext( bn5c_branch2b_id       )
    topology.get( bn5c_branch2b_id       ).addNext( res5c_branch2b_relu_id )
    topology.get( res5c_branch2b_relu_id ).addNext( res5c_branch2c_id      )
    topology.get( res5c_branch2c_id      ).addNext( bn5c_branch2c_id       )
    topology.get( bn5c_branch2c_id       ).addNext( res5c_id               )
    topology.get( res5c_id               ).addNext( res5c_relu_id          )
    topology.get( res5c_relu_id          ).addNext( pool5_id               )
    topology.get( pool5_id               ).addNext( fc1000_id              )
    topology.get( fc1000_id              ).addNext( loss_layer_id          )

    return topology


def main():

    defaultDatasetsPath = os.path.join('.', 'data')
    datasetFileNames = ['train_224x224.blob', 'test_224x224.blob']

    userDatasetsPath = getUserDatasetPath(sys.argv)
    datasetsPath = selectDatasetPathOrExit(defaultDatasetsPath, userDatasetsPath, datasetFileNames, 2)

    # Form path to the training and testing datasets
    trainBlobPath = os.path.join(datasetsPath, datasetFileNames[0])
    testBlobPath = os.path.join(datasetsPath, datasetFileNames[1])

    # Create blob dataset reader for the training dataset (ImageBlobDatasetReader defined in blob_dataset.py)
    trainDatasetReader = ImageBlobDatasetReader(trainBlobPath, batchSize)
    topology = configureNet()

    # Train model (trainClassifier is defined in daal_commons.py)
    predictionModel = trainClassifier(topology, trainDatasetReader)

    # Create blob dataset reader for the testing dataset
    testDatasetReader = ImageBlobDatasetReader(testBlobPath, batchSize)

    # Test model (testClassifier is defined in daal_commons.py)
    top5ErrorRate = testClassifier(predictionModel, testDatasetReader)

    print("Top-5 error = {:.3f}%".format(top5ErrorRate * 100.0))


if __name__ == "__main__":
    main()
