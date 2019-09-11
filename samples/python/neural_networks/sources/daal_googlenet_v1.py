# file: daal_googlenet_v1.py
#===============================================================================
# Copyright 2017-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#
# License:
# http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
# eement/
#===============================================================================

#
# !  Content:
# !    Python example of neural network training and scoring with GoogleNetV1 topology
# !*****************************************************************************
from __future__ import division

import os
import sys

from daal.algorithms.neural_networks import training
from daal.algorithms.neural_networks.layers import (
    convolution2d, relu, lrn, maximum_pooling2d, fullyconnected, dropout, loss,
    pooling2d, average_pooling2d, split, concat
)
from daal.algorithms.neural_networks.initializers import uniform, xavier

from daal_commons import trainClassifier, testClassifier, batchSize
from blob_dataset import ImageBlobDatasetReader
from service import getUserDatasetPath, selectDatasetPathOrExit


def configureLossLayers():

    topology = training.Topology()

    # pooling(loss1/ave_pool): 5x5 + 3x3s
    loss_ave_pool = average_pooling2d.Batch(4)
    loss_ave_pool.parameter.kernelSizes = pooling2d.KernelSizes(5, 5)
    loss_ave_pool.parameter.strides = pooling2d.Strides(3, 3)
    loss_ave_pool.parameter.paddings = pooling2d.Paddings(0, 0)
    loss_ave_pool_id = topology.add(loss_ave_pool)

    # convolution(loss/conv): 1x1@128 + 1x1s
    loss_conv = convolution2d.Batch()
    loss_conv.parameter.nKernels = 128
    loss_conv.parameter.kernelSizes = convolution2d.KernelSizes(1, 1)
    loss_conv.parameter.strides = convolution2d.Strides(1, 1)
    loss_conv.parameter.paddings = convolution2d.Paddings(0, 0)
    loss_conv.parameter.weightsInitializer = xavier.Batch()
    loss_conv.parameter.biasesInitializer = uniform.Batch(0.2, 0.2)
    loss_conv_id = topology.add(loss_conv)

    # relu(loss/relu_conv)
    loss_relu_conv = relu.Batch()
    loss_relu_conv_id = topology.add(loss_relu_conv)

    # fullyconnected(loss/fc): n = 1024
    loss_fc = fullyconnected.Batch(1024)
    loss_fc.parameter.weightsInitializer = xavier.Batch()
    loss_fc.parameter.biasesInitializer = uniform.Batch(0.2, 0.2)
    loss_fc_id = topology.add(loss_fc)

    # relu(loss/relu_fc)
    loss_relu_fc = relu.Batch()
    loss_relu_fc_id = topology.add(loss_relu_fc)

    # dropout(loss/drop_fc): p = 0.7
    loss_drop_fc = dropout.Batch()
    loss_drop_fc.parameter.retainRatio = 0.7
    loss_drop_fc_id = topology.add(loss_drop_fc)

    # fullyconnected(loss/classifier): n = 1000
    loss_classifier = fullyconnected.Batch(1000)
    loss_classifier.parameter.weightsInitializer = xavier.Batch()
    loss_classifier.parameter.biasesInitializer = uniform.Batch(0.0, 0.0)
    loss_classifier_id = topology.add(loss_classifier)

    # softmax + crossentropy loss(loss/loss)
    loss_loss = loss.softmax_cross.Batch()
    loss_loss_id = topology.add(loss_loss)

    topology.get(loss_ave_pool_id  ).addNext(loss_conv_id      )
    topology.get(loss_conv_id      ).addNext(loss_relu_conv_id )
    topology.get(loss_relu_conv_id ).addNext(loss_fc_id        )
    topology.get(loss_fc_id        ).addNext(loss_relu_fc_id   )
    topology.get(loss_relu_fc_id   ).addNext(loss_drop_fc_id   )
    topology.get(loss_drop_fc_id   ).addNext(loss_classifier_id)
    topology.get(loss_classifier_id).addNext(loss_loss_id      )

    return topology


def configureInception(nKernels1, nKernels2, nKernels3, nKernels4, nKernels5, nKernels6, hasLoss=False):

    topology = training.Topology()

    splitSuccessors = 4
    if hasLoss:
        splitSuccessors += 1

    # split
    split_layer = split.Batch(splitSuccessors, splitSuccessors)
    split_id = topology.add(split_layer)

    # convolution(inception/1x1): 1x1@nKernels1 + 1x1s
    inception_1x1 = convolution2d.Batch()
    inception_1x1.parameter.nKernels = nKernels1
    inception_1x1.parameter.kernelSizes = convolution2d.KernelSizes(1, 1)
    inception_1x1.parameter.strides = convolution2d.Strides(1, 1)
    inception_1x1.parameter.weightsInitializer = xavier.Batch()
    inception_1x1.parameter.biasesInitializer = uniform.Batch(0.2, 0.2)
    inception_1x1_id = topology.add(inception_1x1)

    # relu(inception/relu_1x1)
    inception_relu_1x1 = relu.Batch()
    inception_relu_1x1_id = topology.add(inception_relu_1x1)

    # convolution(inception/3x3_reduce): 1x1@nKernels2 + 1x1s
    inception_3x3_reduce = convolution2d.Batch()
    inception_3x3_reduce.parameter.nKernels = nKernels2
    inception_3x3_reduce.parameter.kernelSizes = convolution2d.KernelSizes(1, 1)
    inception_3x3_reduce.parameter.strides = convolution2d.Strides(1, 1)
    inception_3x3_reduce.parameter.weightsInitializer = xavier.Batch()
    inception_3x3_reduce.parameter.biasesInitializer = uniform.Batch(0.2, 0.2)
    inception_3x3_reduce_id = topology.add(inception_3x3_reduce)

    # relu(inception/relu_3x3_reduce)
    inception_relu_3x3_reduce = relu.Batch()
    inception_relu_3x3_reduce_id = topology.add(inception_relu_3x3_reduce)

    # convolution(inception/3x3): 3x3@nKernels3 + 1x1s
    inception_3x3 = convolution2d.Batch()
    inception_3x3.parameter.nKernels = nKernels3
    inception_3x3.parameter.kernelSizes = convolution2d.KernelSizes(3, 3)
    inception_3x3.parameter.strides = convolution2d.Strides(1, 1)
    inception_3x3.parameter.paddings = convolution2d.Paddings(1, 1)
    inception_3x3.parameter.weightsInitializer = xavier.Batch()
    inception_3x3.parameter.biasesInitializer = uniform.Batch(0.2, 0.2)
    inception_3x3_id = topology.add(inception_3x3)

    # relu(inception/relu_3x3)
    inception_relu_3x3 = relu.Batch()
    inception_relu_3x3_id = topology.add(inception_relu_3x3)

    # convolution(inception/5x5_reduce): 1x1@nKernels4 + 1x1s
    inception_5x5_reduce = convolution2d.Batch()
    inception_5x5_reduce.parameter.nKernels = nKernels4
    inception_5x5_reduce.parameter.kernelSizes = convolution2d.KernelSizes(1, 1)
    inception_5x5_reduce.parameter.strides = convolution2d.Strides(1, 1)
    inception_5x5_reduce.parameter.weightsInitializer = xavier.Batch()
    inception_5x5_reduce.parameter.biasesInitializer = uniform.Batch(0.2, 0.2)
    inception_5x5_reduce_id = topology.add(inception_5x5_reduce)

    # relu(inception/relu_5x5_reduce)
    inception_relu_5x5_reduce = relu.Batch()
    inception_relu_5x5_reduce_id = topology.add(inception_relu_5x5_reduce)

    # convolution(inception/5x5): 5x5@nKernels5 + 1x1s
    inception_5x5 = convolution2d.Batch()
    inception_5x5.parameter.nKernels = nKernels5
    inception_5x5.parameter.kernelSizes = convolution2d.KernelSizes(5, 5)
    inception_5x5.parameter.strides = convolution2d.Strides(1, 1)
    inception_5x5.parameter.paddings = convolution2d.Paddings(2, 2)
    inception_5x5.parameter.weightsInitializer = xavier.Batch()
    inception_5x5.parameter.biasesInitializer = uniform.Batch(0.2, 0.2)
    inception_5x5_id = topology.add(inception_5x5)

    # relu(inception/relu_5x5)
    inception_relu_5x5 = relu.Batch()
    inception_relu_5x5_id = topology.add(inception_relu_5x5)

    # pooling(inception/pool): 3x3 + 1x1s
    inception_pool = maximum_pooling2d.Batch(4)
    inception_pool.parameter.kernelSizes = pooling2d.KernelSizes(3, 3)
    inception_pool.parameter.strides = pooling2d.Strides(1, 1)
    inception_pool.parameter.paddings = pooling2d.Paddings(1, 1)
    inception_pool_id = topology.add(inception_pool)

    # convolution(inception/pool_proj): 1x1@nKernels6 + 1x1s
    inception_pool_proj = convolution2d.Batch()
    inception_pool_proj.parameter.nKernels = nKernels6
    inception_pool_proj.parameter.kernelSizes = convolution2d.KernelSizes(1, 1)
    inception_pool_proj.parameter.strides = convolution2d.Strides(1, 1)
    inception_pool_proj.parameter.weightsInitializer = xavier.Batch()
    inception_pool_proj.parameter.biasesInitializer = uniform.Batch(0.2, 0.2)
    inception_pool_proj_id = topology.add(inception_pool_proj)

    # relu(inception/relu_pool_proj)
    inception_relu_pool_proj = relu.Batch()
    inception_relu_pool_proj_id = topology.add(inception_relu_pool_proj)

    # concat(inception/output)
    inception_output = concat.Batch()
    inception_output.parameter.concatDimension = 1
    inception_output_id = topology.add(inception_output)

    topology.get(split_id                    ).addNext(inception_3x3_reduce_id     )
    topology.get(split_id                    ).addNext(inception_pool_id           )
    topology.get(split_id                    ).addNext(inception_1x1_id            )
    topology.get(split_id                    ).addNext(inception_5x5_reduce_id     )
    topology.get(inception_1x1_id            ).addNext(inception_relu_1x1_id       )
    topology.get(inception_relu_1x1_id       ).addNext(inception_output_id         )
    topology.get(inception_3x3_reduce_id     ).addNext(inception_relu_3x3_reduce_id)
    topology.get(inception_relu_3x3_reduce_id).addNext(inception_3x3_id            )
    topology.get(inception_3x3_id            ).addNext(inception_relu_3x3_id       )
    topology.get(inception_relu_3x3_id       ).addNext(inception_output_id         )
    topology.get(inception_5x5_reduce_id     ).addNext(inception_relu_5x5_reduce_id)
    topology.get(inception_relu_5x5_reduce_id).addNext(inception_5x5_id            )
    topology.get(inception_5x5_id            ).addNext(inception_relu_5x5_id       )
    topology.get(inception_relu_5x5_id       ).addNext(inception_output_id         )
    topology.get(inception_pool_id           ).addNext(inception_pool_proj_id      )
    topology.get(inception_pool_proj_id      ).addNext(inception_relu_pool_proj_id )
    topology.get(inception_relu_pool_proj_id ).addNext(inception_output_id         )

    return topology


def configureNet():

    topology = training.Topology()

    # convolution(conv1/7x7_s2): 7x7@64 + 2x2s
    conv1_7x7_s2 = convolution2d.Batch()
    conv1_7x7_s2.parameter.nKernels = 64
    conv1_7x7_s2.parameter.kernelSizes = convolution2d.KernelSizes(7, 7)
    conv1_7x7_s2.parameter.strides = convolution2d.Strides(2, 2)
    conv1_7x7_s2.parameter.paddings = convolution2d.Paddings(3, 3)
    conv1_7x7_s2.parameter.weightsInitializer = xavier.Batch()
    conv1_7x7_s2.parameter.biasesInitializer = uniform.Batch(0.2, 0.2)
    conv1_7x7_s2_id = topology.add(conv1_7x7_s2)

    # relu(conv1/relu_7x7)
    conv1_relu_7x7 = relu.Batch()
    conv1_relu_7x7_id = topology.add(conv1_relu_7x7)

    # pooling(pool1/3x3_s2): 3x3 + 2x2s
    pool1_3x3_s2 = maximum_pooling2d.Batch(4)
    pool1_3x3_s2.parameter.kernelSizes = pooling2d.KernelSizes(3, 3)
    pool1_3x3_s2.parameter.strides = pooling2d.Strides(2, 2)
    pool1_3x3_s2_id = topology.add(pool1_3x3_s2)

    # lrn(pool1/norm1): alpha=0.0001, beta=0.75, local_size=5
    pool1_norm1 = lrn.Batch()
    pool1_norm1.parameter.kappa = 1
    pool1_norm1.parameter.nAdjust = 5
    pool1_norm1.parameter.beta = 0.75
    pool1_norm1.parameter.alpha = 0.0001 / pool1_norm1.parameter.nAdjust
    pool1_norm1_id = topology.add(pool1_norm1)

    # convolution(conv2/3x3_reduce): 1x1@64 + 1x1s
    conv2_3x3_reduce = convolution2d.Batch()
    conv2_3x3_reduce.parameter.nKernels = 64
    conv2_3x3_reduce.parameter.kernelSizes = convolution2d.KernelSizes(1, 1)
    conv2_3x3_reduce.parameter.strides = convolution2d.Strides(1, 1)
    conv2_3x3_reduce.parameter.paddings = convolution2d.Paddings(0, 0)
    conv2_3x3_reduce.parameter.weightsInitializer = xavier.Batch()
    conv2_3x3_reduce.parameter.biasesInitializer = uniform.Batch(0.2, 0.2)
    conv2_3x3_reduce_id = topology.add(conv2_3x3_reduce)

    # convolution(conv2/relu_3x3_reduce)
    conv2_relu_3x3_reduce = relu.Batch()
    conv2_relu_3x3_reduce_id = topology.add(conv2_relu_3x3_reduce)

    # convolution(conv2/3x3): 3x3@192 + 1x1s
    conv2_3x3 = convolution2d.Batch()
    conv2_3x3.parameter.nKernels = 192
    conv2_3x3.parameter.kernelSizes = convolution2d.KernelSizes(3, 3)
    conv2_3x3.parameter.strides = convolution2d.Strides(1, 1)
    conv2_3x3.parameter.paddings = convolution2d.Paddings(1, 1)
    conv2_3x3.parameter.weightsInitializer = xavier.Batch()
    conv2_3x3.parameter.biasesInitializer = uniform.Batch(0.2, 0.2)
    conv2_3x3_id = topology.add(conv2_3x3)

    # relu(conv2/relu_3x3)
    conv2_relu_3x3 = relu.Batch()
    conv2_relu_3x3_id = topology.add(conv2_relu_3x3)

    # lrn(conv2/norm2): alpha=0.0001, beta=0.75, local_size=5
    conv2_norm2 = lrn.Batch()
    conv2_norm2.parameter.kappa = 1
    conv2_norm2.parameter.nAdjust = 5
    conv2_norm2.parameter.beta = 0.75
    conv2_norm2.parameter.alpha = 0.0001 / conv2_norm2.parameter.nAdjust
    conv2_norm2_id = topology.add(conv2_norm2)

    # pooling(pool2/3x3_s2): 3x3 + 1x1s
    pool2_3x3_s2 = maximum_pooling2d.Batch(4)
    pool2_3x3_s2.parameter.kernelSizes = pooling2d.KernelSizes(3, 3)
    pool2_3x3_s2.parameter.strides = pooling2d.Strides(2, 2)
    pool2_3x3_s2_id = topology.add(pool2_3x3_s2)

    # inception module, convolution filters = (64, 96, 128, 16, 32, 32)
    inception1 = configureInception(64, 96, 128, 16, 32, 32)
    inception1_end_id, inception1_start_id = topology.add(inception1)

    # inception module, convolution filters = (128, 128, 192, 32, 96, 64)
    inception2 = configureInception(128, 128, 192, 32, 96, 64)
    inception2_end_id, inception2_start_id = topology.add(inception2)

    # pooling(pool3/3x3_s2): 3x3 + 2x2s
    pool3_3x3_s2 = maximum_pooling2d.Batch(4)
    pool3_3x3_s2.parameter.kernelSizes = pooling2d.KernelSizes(3, 3)
    pool3_3x3_s2.parameter.strides = pooling2d.Strides(2, 2)
    pool3_3x3_s2_id = topology.add(pool3_3x3_s2)

    # inception module, convolution filters = (192, 96, 208, 16, 48, 64)
    inception3 = configureInception(192, 96, 208, 16, 48, 64)
    inception3_end_id, inception3_start_id = topology.add(inception3)

    # inception module, convolution filters = (160, 112, 224, 24, 64, 64)
    inception4 = configureInception(160, 112, 224, 24, 64, 64, True)  # has loss branch
    inception4_end_id, inception4_start_id = topology.add(inception4)

    # loss branch 1
    loss1Branch = configureLossLayers()
    loss1_end_id, loss1_start_id = topology.add(loss1Branch)

    # inception module, convolution filters = (128, 128, 256, 24, 64, 64)
    inception5 = configureInception(128, 128, 256, 24, 64, 64)
    inception5_end_id, inception5_start_id = topology.add(inception5)

    # inception module, convolution filters = (112, 144, 288, 32, 64, 64)
    inception6 = configureInception(112, 144, 288, 32, 64, 64)
    inception6_end_id, inception6_start_id = topology.add(inception6)

    # inception module, convolution filters = (256, 160, 320, 32, 128, 128)
    inception7 = configureInception(256, 160, 320, 32, 128, 128, True)  # has loss branch
    inception7_end_id, inception7_start_id = topology.add(inception7)

    # loss branch 2
    loss2Branch = configureLossLayers()
    loss2_end_id, loss2_start_id = topology.add(loss2Branch)

    # pooling(pool4_3x3_s2): 3x3 + 2x2s
    pool4_3x3_s2 = maximum_pooling2d.Batch(4)
    pool4_3x3_s2.parameter.kernelSizes = pooling2d.KernelSizes(3, 3)
    pool4_3x3_s2.parameter.strides = pooling2d.Strides(2, 2)
    pool4_3x3_s2_id = topology.add(pool4_3x3_s2)

    # inception module, convolution filters = (256, 160, 320, 32, 128, 128)
    inception8 = configureInception(256, 160, 320, 32, 128, 128)
    inception8_end_id, inception8_start_id = topology.add(inception8)

    # inception module, convolution filters = (384, 192, 384, 48, 128, 128)
    inception9 = configureInception(384, 192, 384, 48, 128, 128)
    inception9_end_id, inception9_start_id = topology.add(inception9)

    # pooling(pool5/7x7_s1): 7x7 + 1x1s
    pool5_7x7_s1 = average_pooling2d.Batch(4)
    pool5_7x7_s1.parameter.kernelSizes = pooling2d.KernelSizes(7, 7)
    pool5_7x7_s1.parameter.strides = pooling2d.Strides(1, 1)
    pool5_7x7_s1_id = topology.add(pool5_7x7_s1)

    # dropout(pool5/drop_7x7_s1): p = 0.4
    pool5_drop_7x7_s1 = dropout.Batch()
    pool5_drop_7x7_s1.parameter.retainRatio = 0.4
    pool5_drop_7x7_s1_id = topology.add(pool5_drop_7x7_s1)

    # fullyconnected(loss3/classifier): n = 1000
    loss3_classifier = fullyconnected.Batch(1000)
    loss3_classifier.parameter.weightsInitializer = xavier.Batch()
    loss3_classifier.parameter.biasesInitializer = uniform.Batch(0.0, 0.0)
    loss3_classifier_id = topology.add(loss3_classifier)

    # softmax + crossentropy loss (loss3/loss3)
    loss3_loss3 = loss.softmax_cross.Batch()
    loss3_end_id = topology.add(loss3_loss3)

    topology.get(conv1_7x7_s2_id         ).addNext(conv1_relu_7x7_id       )
    topology.get(conv1_relu_7x7_id       ).addNext(pool1_3x3_s2_id         )
    topology.get(pool1_3x3_s2_id         ).addNext(pool1_norm1_id          )
    topology.get(pool1_norm1_id          ).addNext(conv2_3x3_reduce_id     )
    topology.get(conv2_3x3_reduce_id     ).addNext(conv2_relu_3x3_reduce_id)
    topology.get(conv2_relu_3x3_reduce_id).addNext(conv2_3x3_id            )
    topology.get(conv2_3x3_id            ).addNext(conv2_relu_3x3_id       )
    topology.get(conv2_relu_3x3_id       ).addNext(conv2_norm2_id          )
    topology.get(conv2_norm2_id          ).addNext(pool2_3x3_s2_id         )
    topology.get(pool2_3x3_s2_id         ).addNext(inception1_start_id     )
    topology.get(inception1_end_id       ).addNext(inception2_start_id     )
    topology.get(inception2_end_id       ).addNext(pool3_3x3_s2_id         )
    topology.get(pool3_3x3_s2_id         ).addNext(inception3_start_id     )
    topology.get(inception3_end_id       ).addNext(inception4_start_id     )
    topology.get(inception4_start_id     ).addNext(loss1_start_id          )
    topology.get(inception4_end_id       ).addNext(inception5_start_id     )
    topology.get(inception5_end_id       ).addNext(inception6_start_id     )
    topology.get(inception6_end_id       ).addNext(inception7_start_id     )
    topology.get(inception7_start_id     ).addNext(loss2_start_id          )
    topology.get(inception7_end_id       ).addNext(pool4_3x3_s2_id         )
    topology.get(pool4_3x3_s2_id         ).addNext(inception8_start_id     )
    topology.get(inception8_end_id       ).addNext(inception9_start_id     )
    topology.get(inception9_end_id       ).addNext(pool5_7x7_s1_id         )
    topology.get(pool5_7x7_s1_id         ).addNext(pool5_drop_7x7_s1_id    )
    topology.get(pool5_drop_7x7_s1_id    ).addNext(loss3_classifier_id     )
    topology.get(loss3_classifier_id     ).addNext(loss3_end_id            )

    return topology


def main():
    defaultDatasetsPath = os.path.join('.', 'data')
    datasetFileNames = ['train_224x224.blob', 'test_224x224.blob']

    userDatasetsPath = getUserDatasetPath(sys.argv)
    datasetsPath = selectDatasetPathOrExit(defaultDatasetsPath, userDatasetsPath, datasetFileNames, 2)

    # Form path to the training and testing datasets
    trainBlobPath = os.path.join(datasetsPath, datasetFileNames[0])
    testBlobPath  = os.path.join(datasetsPath, datasetFileNames[1])

    # Create blob dataset reader for the training dataset (ImageBlobDatasetReader defined in blob_dataset.py)
    trainDatasetReader = ImageBlobDatasetReader(trainBlobPath, batchSize)
    topology = configureNet()

    # Train model (trainClassifier is defined in daal_commons.py)
    predictionModel = trainClassifier(topology, trainDatasetReader)

    # Create blob dataset reader for the testing dataset
    testDatasetReader = ImageBlobDatasetReader(testBlobPath, batchSize)

    # Test model (testClassifier is defined in daal_commons.py)
    top5ErrorRate = testClassifier(predictionModel, testDatasetReader)

    print("Top-5 error = {}%".format(top5ErrorRate * 100.0))

if __name__ == '__main__':
    main()
