/* file: daal_resnet_50.h */
/*******************************************************************************
* Copyright 2017-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*
* License:
* http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
* eement/
*******************************************************************************/

#ifndef _DAAL_RESNET_50_H
#define _DAAL_RESNET_50_H

#include "daal_defines.h"

training::TopologyPtr configureNet()
{
    training::TopologyPtr topology(new training::Topology());

    SharedPtr<convolution2d::Batch<float> > conv1(new convolution2d::Batch<float>());
    conv1->parameter.nKernels           = 64;
    conv1->parameter.kernelSizes        = convolution2d::KernelSizes(7, 7);
    conv1->parameter.strides            = convolution2d::Strides(2, 2);
    conv1->parameter.paddings           = convolution2d::Paddings(3, 3);
    conv1->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    conv1->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t conv1_id               = topology->add(conv1);

    SharedPtr<batch_normalization::Batch<float> > bn_conv1(new batch_normalization::Batch<float>());
    const size_t bn_conv1_id = topology->add(bn_conv1);

    SharedPtr<relu::Batch<float> > conv1_relu(new relu::Batch<float>());
    const size_t conv1_relu_id = topology->add(conv1_relu);

    SharedPtr<maximum_pooling2d::Batch<float> > pool1(new maximum_pooling2d::Batch<float>(4));
    pool1->parameter.kernelSizes = pooling2d::KernelSizes(3, 3);
    pool1->parameter.strides     = pooling2d::Strides(2, 2);
    const size_t pool1_id        = topology->add(pool1);

    SharedPtr<split::Batch<float> > pool1_split1(new split::Batch<float>(2, 2));
    const size_t pool1_split1_id = topology->add(pool1_split1);

    SharedPtr<convolution2d::Batch<float> > res2a_branch1(new convolution2d::Batch<float>());
    res2a_branch1->parameter.nKernels           = 256;
    res2a_branch1->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res2a_branch1->parameter.strides            = convolution2d::Strides(1, 1);
    res2a_branch1->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res2a_branch1->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res2a_branch1_id               = topology->add(res2a_branch1);

    SharedPtr<batch_normalization::Batch<float> > bn2a_branch1(new batch_normalization::Batch<float>());
    const size_t bn2a_branch1_id = topology->add(bn2a_branch1);

    SharedPtr<convolution2d::Batch<float> > res2a_branch2a(new convolution2d::Batch<float>());
    res2a_branch2a->parameter.nKernels           = 64;
    res2a_branch2a->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res2a_branch2a->parameter.strides            = convolution2d::Strides(1, 1);
    res2a_branch2a->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res2a_branch2a->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res2a_branch2a_id               = topology->add(res2a_branch2a);

    SharedPtr<batch_normalization::Batch<float> > bn2a_branch2a(new batch_normalization::Batch<float>());
    const size_t bn2a_branch2a_id = topology->add(bn2a_branch2a);

    SharedPtr<relu::Batch<float> > res2a_branch2a_relu(new relu::Batch<float>());
    const size_t res2a_branch2a_relu_id = topology->add(res2a_branch2a_relu);

    SharedPtr<convolution2d::Batch<float> > res2a_branch2b(new convolution2d::Batch<float>());
    res2a_branch2b->parameter.nKernels           = 64;
    res2a_branch2b->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    res2a_branch2b->parameter.strides            = convolution2d::Strides(1, 1);
    res2a_branch2b->parameter.paddings           = convolution2d::Paddings(1, 1);
    res2a_branch2b->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res2a_branch2b->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res2a_branch2b_id               = topology->add(res2a_branch2b);

    SharedPtr<batch_normalization::Batch<float> > bn2a_branch2b(new batch_normalization::Batch<float>());
    const size_t bn2a_branch2b_id = topology->add(bn2a_branch2b);

    SharedPtr<relu::Batch<float> > res2a_branch2b_relu(new relu::Batch<float>());
    const size_t res2a_branch2b_relu_id = topology->add(res2a_branch2b_relu);

    SharedPtr<convolution2d::Batch<float> > res2a_branch2c(new convolution2d::Batch<float>());
    res2a_branch2c->parameter.nKernels           = 256;
    res2a_branch2c->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res2a_branch2c->parameter.strides            = convolution2d::Strides(1, 1);
    res2a_branch2c->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res2a_branch2c->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res2a_branch2c_id               = topology->add(res2a_branch2c);

    SharedPtr<batch_normalization::Batch<float> > bn2a_branch2c(new batch_normalization::Batch<float>());
    const size_t bn2a_branch2c_id = topology->add(bn2a_branch2c);

    SharedPtr<eltwise_sum::Batch<float> > res2a(new eltwise_sum::Batch<float>());
    const size_t res2a_id = topology->add(res2a);

    SharedPtr<relu::Batch<float> > res2a_relu(new relu::Batch<float>());
    const size_t res2a_relu_id = topology->add(res2a_relu);

    SharedPtr<split::Batch<float> > res2a_relu_split2(new split::Batch<float>(2, 2));
    const size_t res2a_relu_split2_id = topology->add(res2a_relu_split2);

    SharedPtr<convolution2d::Batch<float> > res2b_branch2a(new convolution2d::Batch<float>());
    res2b_branch2a->parameter.nKernels           = 64;
    res2b_branch2a->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res2b_branch2a->parameter.strides            = convolution2d::Strides(1, 1);
    res2b_branch2a->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res2b_branch2a->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res2b_branch2a_id               = topology->add(res2b_branch2a);

    SharedPtr<batch_normalization::Batch<float> > bn2b_branch2a(new batch_normalization::Batch<float>());
    const size_t bn2b_branch2a_id = topology->add(bn2b_branch2a);

    SharedPtr<relu::Batch<float> > res2b_branch2a_relu(new relu::Batch<float>());
    const size_t res2b_branch2a_relu_id = topology->add(res2b_branch2a_relu);

    SharedPtr<convolution2d::Batch<float> > res2b_branch2b(new convolution2d::Batch<float>());
    res2b_branch2b->parameter.nKernels           = 64;
    res2b_branch2b->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    res2b_branch2b->parameter.strides            = convolution2d::Strides(1, 1);
    res2b_branch2b->parameter.paddings           = convolution2d::Paddings(1, 1);
    res2b_branch2b->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res2b_branch2b->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res2b_branch2b_id               = topology->add(res2b_branch2b);

    SharedPtr<batch_normalization::Batch<float> > bn2b_branch2b(new batch_normalization::Batch<float>());
    const size_t bn2b_branch2b_id = topology->add(bn2b_branch2b);

    SharedPtr<relu::Batch<float> > res2b_branch2b_relu(new relu::Batch<float>());
    const size_t res2b_branch2b_relu_id = topology->add(res2b_branch2b_relu);

    SharedPtr<convolution2d::Batch<float> > res2b_branch2c(new convolution2d::Batch<float>());
    res2b_branch2c->parameter.nKernels           = 256;
    res2b_branch2c->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res2b_branch2c->parameter.strides            = convolution2d::Strides(1, 1);
    res2b_branch2c->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res2b_branch2c->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res2b_branch2c_id               = topology->add(res2b_branch2c);

    SharedPtr<batch_normalization::Batch<float> > bn2b_branch2c(new batch_normalization::Batch<float>());
    const size_t bn2b_branch2c_id = topology->add(bn2b_branch2c);

    SharedPtr<eltwise_sum::Batch<float> > res2b(new eltwise_sum::Batch<float>());
    const size_t res2b_id = topology->add(res2b);

    SharedPtr<relu::Batch<float> > res2b_relu(new relu::Batch<float>());
    const size_t res2b_relu_id = topology->add(res2b_relu);

    SharedPtr<split::Batch<float> > res2b_relu_split3(new split::Batch<float>(2, 2));
    const size_t res2b_relu_split3_id = topology->add(res2b_relu_split3);

    SharedPtr<convolution2d::Batch<float> > res2c_branch2a(new convolution2d::Batch<float>());
    res2c_branch2a->parameter.nKernels           = 64;
    res2c_branch2a->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res2c_branch2a->parameter.strides            = convolution2d::Strides(1, 1);
    res2c_branch2a->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res2c_branch2a->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res2c_branch2a_id               = topology->add(res2c_branch2a);

    SharedPtr<batch_normalization::Batch<float> > bn2c_branch2a(new batch_normalization::Batch<float>());
    const size_t bn2c_branch2a_id = topology->add(bn2c_branch2a);

    SharedPtr<relu::Batch<float> > res2c_branch2a_relu(new relu::Batch<float>());
    const size_t res2c_branch2a_relu_id = topology->add(res2c_branch2a_relu);

    SharedPtr<convolution2d::Batch<float> > res2c_branch2b(new convolution2d::Batch<float>());
    res2c_branch2b->parameter.nKernels           = 64;
    res2c_branch2b->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    res2c_branch2b->parameter.strides            = convolution2d::Strides(1, 1);
    res2c_branch2b->parameter.paddings           = convolution2d::Paddings(1, 1);
    res2c_branch2b->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res2c_branch2b->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res2c_branch2b_id               = topology->add(res2c_branch2b);

    SharedPtr<batch_normalization::Batch<float> > bn2c_branch2b(new batch_normalization::Batch<float>());
    const size_t bn2c_branch2b_id = topology->add(bn2c_branch2b);

    SharedPtr<relu::Batch<float> > res2c_branch2b_relu(new relu::Batch<float>());
    const size_t res2c_branch2b_relu_id = topology->add(res2c_branch2b_relu);

    SharedPtr<convolution2d::Batch<float> > res2c_branch2c(new convolution2d::Batch<float>());
    res2c_branch2c->parameter.nKernels           = 256;
    res2c_branch2c->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res2c_branch2c->parameter.strides            = convolution2d::Strides(1, 1);
    res2c_branch2c->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res2c_branch2c->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res2c_branch2c_id               = topology->add(res2c_branch2c);

    SharedPtr<batch_normalization::Batch<float> > bn2c_branch2c(new batch_normalization::Batch<float>());
    const size_t bn2c_branch2c_id = topology->add(bn2c_branch2c);

    SharedPtr<eltwise_sum::Batch<float> > res2c(new eltwise_sum::Batch<float>());
    const size_t res2c_id = topology->add(res2c);

    SharedPtr<relu::Batch<float> > res2c_relu(new relu::Batch<float>());
    const size_t res2c_relu_id = topology->add(res2c_relu);

    SharedPtr<split::Batch<float> > res2c_relu_split4(new split::Batch<float>(2, 2));
    const size_t res2c_relu_split4_id = topology->add(res2c_relu_split4);

    SharedPtr<convolution2d::Batch<float> > res3a_branch1(new convolution2d::Batch<float>());
    res3a_branch1->parameter.nKernels           = 512;
    res3a_branch1->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res3a_branch1->parameter.strides            = convolution2d::Strides(2, 2);
    res3a_branch1->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res3a_branch1->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res3a_branch1_id               = topology->add(res3a_branch1);

    SharedPtr<batch_normalization::Batch<float> > bn3a_branch1(new batch_normalization::Batch<float>());
    const size_t bn3a_branch1_id = topology->add(bn3a_branch1);

    SharedPtr<convolution2d::Batch<float> > res3a_branch2a(new convolution2d::Batch<float>());
    res3a_branch2a->parameter.nKernels           = 128;
    res3a_branch2a->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res3a_branch2a->parameter.strides            = convolution2d::Strides(2, 2);
    res3a_branch2a->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res3a_branch2a->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res3a_branch2a_id               = topology->add(res3a_branch2a);

    SharedPtr<batch_normalization::Batch<float> > bn3a_branch2a(new batch_normalization::Batch<float>());
    const size_t bn3a_branch2a_id = topology->add(bn3a_branch2a);

    SharedPtr<relu::Batch<float> > res3a_branch2a_relu(new relu::Batch<float>());
    const size_t res3a_branch2a_relu_id = topology->add(res3a_branch2a_relu);

    SharedPtr<convolution2d::Batch<float> > res3a_branch2b(new convolution2d::Batch<float>());
    res3a_branch2b->parameter.nKernels           = 128;
    res3a_branch2b->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    res3a_branch2b->parameter.strides            = convolution2d::Strides(1, 1);
    res3a_branch2b->parameter.paddings           = convolution2d::Paddings(1, 1);
    res3a_branch2b->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res3a_branch2b->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res3a_branch2b_id               = topology->add(res3a_branch2b);

    SharedPtr<batch_normalization::Batch<float> > bn3a_branch2b(new batch_normalization::Batch<float>());
    const size_t bn3a_branch2b_id = topology->add(bn3a_branch2b);

    SharedPtr<relu::Batch<float> > res3a_branch2b_relu(new relu::Batch<float>());
    const size_t res3a_branch2b_relu_id = topology->add(res3a_branch2b_relu);

    SharedPtr<convolution2d::Batch<float> > res3a_branch2c(new convolution2d::Batch<float>());
    res3a_branch2c->parameter.nKernels           = 512;
    res3a_branch2c->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res3a_branch2c->parameter.strides            = convolution2d::Strides(1, 1);
    res3a_branch2c->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res3a_branch2c->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res3a_branch2c_id               = topology->add(res3a_branch2c);

    SharedPtr<batch_normalization::Batch<float> > bn3a_branch2c(new batch_normalization::Batch<float>());
    const size_t bn3a_branch2c_id = topology->add(bn3a_branch2c);

    SharedPtr<eltwise_sum::Batch<float> > res3a(new eltwise_sum::Batch<float>());
    const size_t res3a_id = topology->add(res3a);

    SharedPtr<relu::Batch<float> > res3a_relu(new relu::Batch<float>());
    const size_t res3a_relu_id = topology->add(res3a_relu);

    SharedPtr<split::Batch<float> > res3a_relu_split5(new split::Batch<float>(2, 2));
    const size_t res3a_relu_split5_id = topology->add(res3a_relu_split5);

    SharedPtr<convolution2d::Batch<float> > res3b_branch2a(new convolution2d::Batch<float>());
    res3b_branch2a->parameter.nKernels           = 128;
    res3b_branch2a->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res3b_branch2a->parameter.strides            = convolution2d::Strides(1, 1);
    res3b_branch2a->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res3b_branch2a->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res3b_branch2a_id               = topology->add(res3b_branch2a);

    SharedPtr<batch_normalization::Batch<float> > bn3b_branch2a(new batch_normalization::Batch<float>());
    const size_t bn3b_branch2a_id = topology->add(bn3b_branch2a);

    SharedPtr<relu::Batch<float> > res3b_branch2a_relu(new relu::Batch<float>());
    const size_t res3b_branch2a_relu_id = topology->add(res3b_branch2a_relu);

    SharedPtr<convolution2d::Batch<float> > res3b_branch2b(new convolution2d::Batch<float>());
    res3b_branch2b->parameter.nKernels           = 128;
    res3b_branch2b->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    res3b_branch2b->parameter.strides            = convolution2d::Strides(1, 1);
    res3b_branch2b->parameter.paddings           = convolution2d::Paddings(1, 1);
    res3b_branch2b->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res3b_branch2b->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res3b_branch2b_id               = topology->add(res3b_branch2b);

    SharedPtr<batch_normalization::Batch<float> > bn3b_branch2b(new batch_normalization::Batch<float>());
    const size_t bn3b_branch2b_id = topology->add(bn3b_branch2b);

    SharedPtr<relu::Batch<float> > res3b_branch2b_relu(new relu::Batch<float>());
    const size_t res3b_branch2b_relu_id = topology->add(res3b_branch2b_relu);

    SharedPtr<convolution2d::Batch<float> > res3b_branch2c(new convolution2d::Batch<float>());
    res3b_branch2c->parameter.nKernels           = 512;
    res3b_branch2c->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res3b_branch2c->parameter.strides            = convolution2d::Strides(1, 1);
    res3b_branch2c->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res3b_branch2c->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res3b_branch2c_id               = topology->add(res3b_branch2c);

    SharedPtr<batch_normalization::Batch<float> > bn3b_branch2c(new batch_normalization::Batch<float>());
    const size_t bn3b_branch2c_id = topology->add(bn3b_branch2c);

    SharedPtr<eltwise_sum::Batch<float> > res3b(new eltwise_sum::Batch<float>());
    const size_t res3b_id = topology->add(res3b);

    SharedPtr<relu::Batch<float> > res3b_relu(new relu::Batch<float>());
    const size_t res3b_relu_id = topology->add(res3b_relu);

    SharedPtr<split::Batch<float> > res3b_relu_split6(new split::Batch<float>(2, 2));
    const size_t res3b_relu_split6_id = topology->add(res3b_relu_split6);

    SharedPtr<convolution2d::Batch<float> > res3c_branch2a(new convolution2d::Batch<float>());
    res3c_branch2a->parameter.nKernels           = 128;
    res3c_branch2a->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res3c_branch2a->parameter.strides            = convolution2d::Strides(1, 1);
    res3c_branch2a->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res3c_branch2a->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res3c_branch2a_id               = topology->add(res3c_branch2a);

    SharedPtr<batch_normalization::Batch<float> > bn3c_branch2a(new batch_normalization::Batch<float>());
    const size_t bn3c_branch2a_id = topology->add(bn3c_branch2a);

    SharedPtr<relu::Batch<float> > res3c_branch2a_relu(new relu::Batch<float>());
    const size_t res3c_branch2a_relu_id = topology->add(res3c_branch2a_relu);

    SharedPtr<convolution2d::Batch<float> > res3c_branch2b(new convolution2d::Batch<float>());
    res3c_branch2b->parameter.nKernels           = 128;
    res3c_branch2b->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    res3c_branch2b->parameter.strides            = convolution2d::Strides(1, 1);
    res3c_branch2b->parameter.paddings           = convolution2d::Paddings(1, 1);
    res3c_branch2b->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res3c_branch2b->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res3c_branch2b_id               = topology->add(res3c_branch2b);

    SharedPtr<batch_normalization::Batch<float> > bn3c_branch2b(new batch_normalization::Batch<float>());
    const size_t bn3c_branch2b_id = topology->add(bn3c_branch2b);

    SharedPtr<relu::Batch<float> > res3c_branch2b_relu(new relu::Batch<float>());
    const size_t res3c_branch2b_relu_id = topology->add(res3c_branch2b_relu);

    SharedPtr<convolution2d::Batch<float> > res3c_branch2c(new convolution2d::Batch<float>());
    res3c_branch2c->parameter.nKernels           = 512;
    res3c_branch2c->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res3c_branch2c->parameter.strides            = convolution2d::Strides(1, 1);
    res3c_branch2c->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res3c_branch2c->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res3c_branch2c_id               = topology->add(res3c_branch2c);

    SharedPtr<batch_normalization::Batch<float> > bn3c_branch2c(new batch_normalization::Batch<float>());
    const size_t bn3c_branch2c_id = topology->add(bn3c_branch2c);

    SharedPtr<eltwise_sum::Batch<float> > res3c(new eltwise_sum::Batch<float>());
    const size_t res3c_id = topology->add(res3c);

    SharedPtr<relu::Batch<float> > res3c_relu(new relu::Batch<float>());
    const size_t res3c_relu_id = topology->add(res3c_relu);

    SharedPtr<split::Batch<float> > res3c_relu_split7(new split::Batch<float>(2, 2));
    const size_t res3c_relu_split7_id = topology->add(res3c_relu_split7);

    SharedPtr<convolution2d::Batch<float> > res3d_branch2a(new convolution2d::Batch<float>());
    res3d_branch2a->parameter.nKernels           = 128;
    res3d_branch2a->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res3d_branch2a->parameter.strides            = convolution2d::Strides(1, 1);
    res3d_branch2a->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res3d_branch2a->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res3d_branch2a_id               = topology->add(res3d_branch2a);

    SharedPtr<batch_normalization::Batch<float> > bn3d_branch2a(new batch_normalization::Batch<float>());
    const size_t bn3d_branch2a_id = topology->add(bn3d_branch2a);

    SharedPtr<relu::Batch<float> > res3d_branch2a_relu(new relu::Batch<float>());
    const size_t res3d_branch2a_relu_id = topology->add(res3d_branch2a_relu);

    SharedPtr<convolution2d::Batch<float> > res3d_branch2b(new convolution2d::Batch<float>());
    res3d_branch2b->parameter.nKernels           = 128;
    res3d_branch2b->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    res3d_branch2b->parameter.strides            = convolution2d::Strides(1, 1);
    res3d_branch2b->parameter.paddings           = convolution2d::Paddings(1, 1);
    res3d_branch2b->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res3d_branch2b->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res3d_branch2b_id               = topology->add(res3d_branch2b);

    SharedPtr<batch_normalization::Batch<float> > bn3d_branch2b(new batch_normalization::Batch<float>());
    const size_t bn3d_branch2b_id = topology->add(bn3d_branch2b);

    SharedPtr<relu::Batch<float> > res3d_branch2b_relu(new relu::Batch<float>());
    const size_t res3d_branch2b_relu_id = topology->add(res3d_branch2b_relu);

    SharedPtr<convolution2d::Batch<float> > res3d_branch2c(new convolution2d::Batch<float>());
    res3d_branch2c->parameter.nKernels           = 512;
    res3d_branch2c->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res3d_branch2c->parameter.strides            = convolution2d::Strides(1, 1);
    res3d_branch2c->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res3d_branch2c->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res3d_branch2c_id               = topology->add(res3d_branch2c);

    SharedPtr<batch_normalization::Batch<float> > bn3d_branch2c(new batch_normalization::Batch<float>());
    const size_t bn3d_branch2c_id = topology->add(bn3d_branch2c);

    SharedPtr<eltwise_sum::Batch<float> > res3d(new eltwise_sum::Batch<float>());
    const size_t res3d_id = topology->add(res3d);

    SharedPtr<relu::Batch<float> > res3d_relu(new relu::Batch<float>());
    const size_t res3d_relu_id = topology->add(res3d_relu);

    SharedPtr<split::Batch<float> > res3d_relu_split8(new split::Batch<float>(2, 2));
    const size_t res3d_relu_split8_id = topology->add(res3d_relu_split8);

    SharedPtr<convolution2d::Batch<float> > res4a_branch1(new convolution2d::Batch<float>());
    res4a_branch1->parameter.nKernels           = 1024;
    res4a_branch1->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res4a_branch1->parameter.strides            = convolution2d::Strides(2, 2);
    res4a_branch1->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4a_branch1->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4a_branch1_id               = topology->add(res4a_branch1);

    SharedPtr<batch_normalization::Batch<float> > bn4a_branch1(new batch_normalization::Batch<float>());
    const size_t bn4a_branch1_id = topology->add(bn4a_branch1);

    SharedPtr<convolution2d::Batch<float> > res4a_branch2a(new convolution2d::Batch<float>());
    res4a_branch2a->parameter.nKernels           = 256;
    res4a_branch2a->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res4a_branch2a->parameter.strides            = convolution2d::Strides(2, 2);
    res4a_branch2a->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4a_branch2a->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4a_branch2a_id               = topology->add(res4a_branch2a);

    SharedPtr<batch_normalization::Batch<float> > bn4a_branch2a(new batch_normalization::Batch<float>());
    const size_t bn4a_branch2a_id = topology->add(bn4a_branch2a);

    SharedPtr<relu::Batch<float> > res4a_branch2a_relu(new relu::Batch<float>());
    const size_t res4a_branch2a_relu_id = topology->add(res4a_branch2a_relu);

    SharedPtr<convolution2d::Batch<float> > res4a_branch2b(new convolution2d::Batch<float>());
    res4a_branch2b->parameter.nKernels           = 256;
    res4a_branch2b->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    res4a_branch2b->parameter.strides            = convolution2d::Strides(1, 1);
    res4a_branch2b->parameter.paddings           = convolution2d::Paddings(1, 1);
    res4a_branch2b->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4a_branch2b->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4a_branch2b_id               = topology->add(res4a_branch2b);

    SharedPtr<batch_normalization::Batch<float> > bn4a_branch2b(new batch_normalization::Batch<float>());
    const size_t bn4a_branch2b_id = topology->add(bn4a_branch2b);

    SharedPtr<relu::Batch<float> > res4a_branch2b_relu(new relu::Batch<float>());
    const size_t res4a_branch2b_relu_id = topology->add(res4a_branch2b_relu);

    SharedPtr<convolution2d::Batch<float> > res4a_branch2c(new convolution2d::Batch<float>());
    res4a_branch2c->parameter.nKernels           = 1024;
    res4a_branch2c->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res4a_branch2c->parameter.strides            = convolution2d::Strides(1, 1);
    res4a_branch2c->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4a_branch2c->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4a_branch2c_id               = topology->add(res4a_branch2c);

    SharedPtr<batch_normalization::Batch<float> > bn4a_branch2c(new batch_normalization::Batch<float>());
    const size_t bn4a_branch2c_id = topology->add(bn4a_branch2c);

    SharedPtr<eltwise_sum::Batch<float> > res4a(new eltwise_sum::Batch<float>());
    const size_t res4a_id = topology->add(res4a);

    SharedPtr<relu::Batch<float> > res4a_relu(new relu::Batch<float>());
    const size_t res4a_relu_id = topology->add(res4a_relu);

    SharedPtr<split::Batch<float> > res4a_relu_split9(new split::Batch<float>(2, 2));
    const size_t res4a_relu_split9_id = topology->add(res4a_relu_split9);

    SharedPtr<convolution2d::Batch<float> > res4b_branch2a(new convolution2d::Batch<float>());
    res4b_branch2a->parameter.nKernels           = 256;
    res4b_branch2a->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res4b_branch2a->parameter.strides            = convolution2d::Strides(1, 1);
    res4b_branch2a->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4b_branch2a->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4b_branch2a_id               = topology->add(res4b_branch2a);

    SharedPtr<batch_normalization::Batch<float> > bn4b_branch2a(new batch_normalization::Batch<float>());
    const size_t bn4b_branch2a_id = topology->add(bn4b_branch2a);

    SharedPtr<relu::Batch<float> > res4b_branch2a_relu(new relu::Batch<float>());
    const size_t res4b_branch2a_relu_id = topology->add(res4b_branch2a_relu);

    SharedPtr<convolution2d::Batch<float> > res4b_branch2b(new convolution2d::Batch<float>());
    res4b_branch2b->parameter.nKernels           = 256;
    res4b_branch2b->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    res4b_branch2b->parameter.strides            = convolution2d::Strides(1, 1);
    res4b_branch2b->parameter.paddings           = convolution2d::Paddings(1, 1);
    res4b_branch2b->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4b_branch2b->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4b_branch2b_id               = topology->add(res4b_branch2b);

    SharedPtr<batch_normalization::Batch<float> > bn4b_branch2b(new batch_normalization::Batch<float>());
    const size_t bn4b_branch2b_id = topology->add(bn4b_branch2b);

    SharedPtr<relu::Batch<float> > res4b_branch2b_relu(new relu::Batch<float>());
    const size_t res4b_branch2b_relu_id = topology->add(res4b_branch2b_relu);

    SharedPtr<convolution2d::Batch<float> > res4b_branch2c(new convolution2d::Batch<float>());
    res4b_branch2c->parameter.nKernels           = 1024;
    res4b_branch2c->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res4b_branch2c->parameter.strides            = convolution2d::Strides(1, 1);
    res4b_branch2c->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4b_branch2c->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4b_branch2c_id               = topology->add(res4b_branch2c);

    SharedPtr<batch_normalization::Batch<float> > bn4b_branch2c(new batch_normalization::Batch<float>());
    const size_t bn4b_branch2c_id = topology->add(bn4b_branch2c);

    SharedPtr<eltwise_sum::Batch<float> > res4b(new eltwise_sum::Batch<float>());
    const size_t res4b_id = topology->add(res4b);

    SharedPtr<relu::Batch<float> > res4b_relu(new relu::Batch<float>());
    const size_t res4b_relu_id = topology->add(res4b_relu);

    SharedPtr<split::Batch<float> > res4b_relu_split10(new split::Batch<float>(2, 2));
    const size_t res4b_relu_split10_id = topology->add(res4b_relu_split10);

    SharedPtr<convolution2d::Batch<float> > res4c_branch2a(new convolution2d::Batch<float>());
    res4c_branch2a->parameter.nKernels           = 256;
    res4c_branch2a->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res4c_branch2a->parameter.strides            = convolution2d::Strides(1, 1);
    res4c_branch2a->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4c_branch2a->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4c_branch2a_id               = topology->add(res4c_branch2a);

    SharedPtr<batch_normalization::Batch<float> > bn4c_branch2a(new batch_normalization::Batch<float>());
    const size_t bn4c_branch2a_id = topology->add(bn4c_branch2a);

    SharedPtr<relu::Batch<float> > res4c_branch2a_relu(new relu::Batch<float>());
    const size_t res4c_branch2a_relu_id = topology->add(res4c_branch2a_relu);

    SharedPtr<convolution2d::Batch<float> > res4c_branch2b(new convolution2d::Batch<float>());
    res4c_branch2b->parameter.nKernels           = 256;
    res4c_branch2b->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    res4c_branch2b->parameter.strides            = convolution2d::Strides(1, 1);
    res4c_branch2b->parameter.paddings           = convolution2d::Paddings(1, 1);
    res4c_branch2b->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4c_branch2b->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4c_branch2b_id               = topology->add(res4c_branch2b);

    SharedPtr<batch_normalization::Batch<float> > bn4c_branch2b(new batch_normalization::Batch<float>());
    const size_t bn4c_branch2b_id = topology->add(bn4c_branch2b);

    SharedPtr<relu::Batch<float> > res4c_branch2b_relu(new relu::Batch<float>());
    const size_t res4c_branch2b_relu_id = topology->add(res4c_branch2b_relu);

    SharedPtr<convolution2d::Batch<float> > res4c_branch2c(new convolution2d::Batch<float>());
    res4c_branch2c->parameter.nKernels           = 1024;
    res4c_branch2c->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res4c_branch2c->parameter.strides            = convolution2d::Strides(1, 1);
    res4c_branch2c->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4c_branch2c->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4c_branch2c_id               = topology->add(res4c_branch2c);

    SharedPtr<batch_normalization::Batch<float> > bn4c_branch2c(new batch_normalization::Batch<float>());
    const size_t bn4c_branch2c_id = topology->add(bn4c_branch2c);

    SharedPtr<eltwise_sum::Batch<float> > res4c(new eltwise_sum::Batch<float>());
    const size_t res4c_id = topology->add(res4c);

    SharedPtr<relu::Batch<float> > res4c_relu(new relu::Batch<float>());
    const size_t res4c_relu_id = topology->add(res4c_relu);

    SharedPtr<split::Batch<float> > res4c_relu_split11(new split::Batch<float>(2, 2));
    const size_t res4c_relu_split11_id = topology->add(res4c_relu_split11);

    SharedPtr<convolution2d::Batch<float> > res4d_branch2a(new convolution2d::Batch<float>());
    res4d_branch2a->parameter.nKernels           = 256;
    res4d_branch2a->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res4d_branch2a->parameter.strides            = convolution2d::Strides(1, 1);
    res4d_branch2a->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4d_branch2a->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4d_branch2a_id               = topology->add(res4d_branch2a);

    SharedPtr<batch_normalization::Batch<float> > bn4d_branch2a(new batch_normalization::Batch<float>());
    const size_t bn4d_branch2a_id = topology->add(bn4d_branch2a);

    SharedPtr<relu::Batch<float> > res4d_branch2a_relu(new relu::Batch<float>());
    const size_t res4d_branch2a_relu_id = topology->add(res4d_branch2a_relu);

    SharedPtr<convolution2d::Batch<float> > res4d_branch2b(new convolution2d::Batch<float>());
    res4d_branch2b->parameter.nKernels           = 256;
    res4d_branch2b->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    res4d_branch2b->parameter.strides            = convolution2d::Strides(1, 1);
    res4d_branch2b->parameter.paddings           = convolution2d::Paddings(1, 1);
    res4d_branch2b->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4d_branch2b->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4d_branch2b_id               = topology->add(res4d_branch2b);

    SharedPtr<batch_normalization::Batch<float> > bn4d_branch2b(new batch_normalization::Batch<float>());
    const size_t bn4d_branch2b_id = topology->add(bn4d_branch2b);

    SharedPtr<relu::Batch<float> > res4d_branch2b_relu(new relu::Batch<float>());
    const size_t res4d_branch2b_relu_id = topology->add(res4d_branch2b_relu);

    SharedPtr<convolution2d::Batch<float> > res4d_branch2c(new convolution2d::Batch<float>());
    res4d_branch2c->parameter.nKernels           = 1024;
    res4d_branch2c->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res4d_branch2c->parameter.strides            = convolution2d::Strides(1, 1);
    res4d_branch2c->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4d_branch2c->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4d_branch2c_id               = topology->add(res4d_branch2c);

    SharedPtr<batch_normalization::Batch<float> > bn4d_branch2c(new batch_normalization::Batch<float>());
    const size_t bn4d_branch2c_id = topology->add(bn4d_branch2c);

    SharedPtr<eltwise_sum::Batch<float> > res4d(new eltwise_sum::Batch<float>());
    const size_t res4d_id = topology->add(res4d);

    SharedPtr<relu::Batch<float> > res4d_relu(new relu::Batch<float>());
    const size_t res4d_relu_id = topology->add(res4d_relu);

    SharedPtr<split::Batch<float> > res4d_relu_split12(new split::Batch<float>(2, 2));
    const size_t res4d_relu_split12_id = topology->add(res4d_relu_split12);

    SharedPtr<convolution2d::Batch<float> > res4e_branch2a(new convolution2d::Batch<float>());
    res4e_branch2a->parameter.nKernels           = 256;
    res4e_branch2a->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res4e_branch2a->parameter.strides            = convolution2d::Strides(1, 1);
    res4e_branch2a->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4e_branch2a->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4e_branch2a_id               = topology->add(res4e_branch2a);

    SharedPtr<batch_normalization::Batch<float> > bn4e_branch2a(new batch_normalization::Batch<float>());
    const size_t bn4e_branch2a_id = topology->add(bn4e_branch2a);

    SharedPtr<relu::Batch<float> > res4e_branch2a_relu(new relu::Batch<float>());
    const size_t res4e_branch2a_relu_id = topology->add(res4e_branch2a_relu);

    SharedPtr<convolution2d::Batch<float> > res4e_branch2b(new convolution2d::Batch<float>());
    res4e_branch2b->parameter.nKernels           = 256;
    res4e_branch2b->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    res4e_branch2b->parameter.strides            = convolution2d::Strides(1, 1);
    res4e_branch2b->parameter.paddings           = convolution2d::Paddings(1, 1);
    res4e_branch2b->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4e_branch2b->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4e_branch2b_id               = topology->add(res4e_branch2b);

    SharedPtr<batch_normalization::Batch<float> > bn4e_branch2b(new batch_normalization::Batch<float>());
    const size_t bn4e_branch2b_id = topology->add(bn4e_branch2b);

    SharedPtr<relu::Batch<float> > res4e_branch2b_relu(new relu::Batch<float>());
    const size_t res4e_branch2b_relu_id = topology->add(res4e_branch2b_relu);

    SharedPtr<convolution2d::Batch<float> > res4e_branch2c(new convolution2d::Batch<float>());
    res4e_branch2c->parameter.nKernels           = 1024;
    res4e_branch2c->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res4e_branch2c->parameter.strides            = convolution2d::Strides(1, 1);
    res4e_branch2c->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4e_branch2c->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4e_branch2c_id               = topology->add(res4e_branch2c);

    SharedPtr<batch_normalization::Batch<float> > bn4e_branch2c(new batch_normalization::Batch<float>());
    const size_t bn4e_branch2c_id = topology->add(bn4e_branch2c);

    SharedPtr<eltwise_sum::Batch<float> > res4e(new eltwise_sum::Batch<float>());
    const size_t res4e_id = topology->add(res4e);

    SharedPtr<relu::Batch<float> > res4e_relu(new relu::Batch<float>());
    const size_t res4e_relu_id = topology->add(res4e_relu);

    SharedPtr<split::Batch<float> > res4e_relu_split13(new split::Batch<float>(2, 2));
    const size_t res4e_relu_split13_id = topology->add(res4e_relu_split13);

    SharedPtr<convolution2d::Batch<float> > res4f_branch2a(new convolution2d::Batch<float>());
    res4f_branch2a->parameter.nKernels           = 256;
    res4f_branch2a->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res4f_branch2a->parameter.strides            = convolution2d::Strides(1, 1);
    res4f_branch2a->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4f_branch2a->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4f_branch2a_id               = topology->add(res4f_branch2a);

    SharedPtr<batch_normalization::Batch<float> > bn4f_branch2a(new batch_normalization::Batch<float>());
    const size_t bn4f_branch2a_id = topology->add(bn4f_branch2a);

    SharedPtr<relu::Batch<float> > res4f_branch2a_relu(new relu::Batch<float>());
    const size_t res4f_branch2a_relu_id = topology->add(res4f_branch2a_relu);

    SharedPtr<convolution2d::Batch<float> > res4f_branch2b(new convolution2d::Batch<float>());
    res4f_branch2b->parameter.nKernels           = 256;
    res4f_branch2b->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    res4f_branch2b->parameter.strides            = convolution2d::Strides(1, 1);
    res4f_branch2b->parameter.paddings           = convolution2d::Paddings(1, 1);
    res4f_branch2b->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4f_branch2b->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4f_branch2b_id               = topology->add(res4f_branch2b);

    SharedPtr<batch_normalization::Batch<float> > bn4f_branch2b(new batch_normalization::Batch<float>());
    const size_t bn4f_branch2b_id = topology->add(bn4f_branch2b);

    SharedPtr<relu::Batch<float> > res4f_branch2b_relu(new relu::Batch<float>());
    const size_t res4f_branch2b_relu_id = topology->add(res4f_branch2b_relu);

    SharedPtr<convolution2d::Batch<float> > res4f_branch2c(new convolution2d::Batch<float>());
    res4f_branch2c->parameter.nKernels           = 1024;
    res4f_branch2c->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res4f_branch2c->parameter.strides            = convolution2d::Strides(1, 1);
    res4f_branch2c->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res4f_branch2c->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res4f_branch2c_id               = topology->add(res4f_branch2c);

    SharedPtr<batch_normalization::Batch<float> > bn4f_branch2c(new batch_normalization::Batch<float>());
    const size_t bn4f_branch2c_id = topology->add(bn4f_branch2c);

    SharedPtr<eltwise_sum::Batch<float> > res4f(new eltwise_sum::Batch<float>());
    const size_t res4f_id = topology->add(res4f);

    SharedPtr<relu::Batch<float> > res4f_relu(new relu::Batch<float>());
    const size_t res4f_relu_id = topology->add(res4f_relu);

    SharedPtr<split::Batch<float> > res4f_relu_split14(new split::Batch<float>(2, 2));
    const size_t res4f_relu_split14_id = topology->add(res4f_relu_split14);

    SharedPtr<convolution2d::Batch<float> > res5a_branch1(new convolution2d::Batch<float>());
    res5a_branch1->parameter.nKernels           = 2048;
    res5a_branch1->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res5a_branch1->parameter.strides            = convolution2d::Strides(2, 2);
    res5a_branch1->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res5a_branch1->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res5a_branch1_id               = topology->add(res5a_branch1);

    SharedPtr<batch_normalization::Batch<float> > bn5a_branch1(new batch_normalization::Batch<float>());
    const size_t bn5a_branch1_id = topology->add(bn5a_branch1);

    SharedPtr<convolution2d::Batch<float> > res5a_branch2a(new convolution2d::Batch<float>());
    res5a_branch2a->parameter.nKernels           = 512;
    res5a_branch2a->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res5a_branch2a->parameter.strides            = convolution2d::Strides(2, 2);
    res5a_branch2a->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res5a_branch2a->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res5a_branch2a_id               = topology->add(res5a_branch2a);

    SharedPtr<batch_normalization::Batch<float> > bn5a_branch2a(new batch_normalization::Batch<float>());
    const size_t bn5a_branch2a_id = topology->add(bn5a_branch2a);

    SharedPtr<relu::Batch<float> > res5a_branch2a_relu(new relu::Batch<float>());
    const size_t res5a_branch2a_relu_id = topology->add(res5a_branch2a_relu);

    SharedPtr<convolution2d::Batch<float> > res5a_branch2b(new convolution2d::Batch<float>());
    res5a_branch2b->parameter.nKernels           = 512;
    res5a_branch2b->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    res5a_branch2b->parameter.strides            = convolution2d::Strides(1, 1);
    res5a_branch2b->parameter.paddings           = convolution2d::Paddings(1, 1);
    res5a_branch2b->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res5a_branch2b->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res5a_branch2b_id               = topology->add(res5a_branch2b);

    SharedPtr<batch_normalization::Batch<float> > bn5a_branch2b(new batch_normalization::Batch<float>());
    const size_t bn5a_branch2b_id = topology->add(bn5a_branch2b);

    SharedPtr<relu::Batch<float> > res5a_branch2b_relu(new relu::Batch<float>());
    const size_t res5a_branch2b_relu_id = topology->add(res5a_branch2b_relu);

    SharedPtr<convolution2d::Batch<float> > res5a_branch2c(new convolution2d::Batch<float>());
    res5a_branch2c->parameter.nKernels           = 2048;
    res5a_branch2c->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res5a_branch2c->parameter.strides            = convolution2d::Strides(1, 1);
    res5a_branch2c->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res5a_branch2c->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res5a_branch2c_id               = topology->add(res5a_branch2c);

    SharedPtr<batch_normalization::Batch<float> > bn5a_branch2c(new batch_normalization::Batch<float>());
    const size_t bn5a_branch2c_id = topology->add(bn5a_branch2c);

    SharedPtr<eltwise_sum::Batch<float> > res5a(new eltwise_sum::Batch<float>());
    const size_t res5a_id = topology->add(res5a);

    SharedPtr<relu::Batch<float> > res5a_relu(new relu::Batch<float>());
    const size_t res5a_relu_id = topology->add(res5a_relu);

    SharedPtr<split::Batch<float> > res5a_relu_split15(new split::Batch<float>(2, 2));
    const size_t res5a_relu_split15_id = topology->add(res5a_relu_split15);

    SharedPtr<convolution2d::Batch<float> > res5b_branch2a(new convolution2d::Batch<float>());
    res5b_branch2a->parameter.nKernels           = 512;
    res5b_branch2a->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res5b_branch2a->parameter.strides            = convolution2d::Strides(1, 1);
    res5b_branch2a->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res5b_branch2a->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res5b_branch2a_id               = topology->add(res5b_branch2a);

    SharedPtr<batch_normalization::Batch<float> > bn5b_branch2a(new batch_normalization::Batch<float>());
    const size_t bn5b_branch2a_id = topology->add(bn5b_branch2a);

    SharedPtr<relu::Batch<float> > res5b_branch2a_relu(new relu::Batch<float>());
    const size_t res5b_branch2a_relu_id = topology->add(res5b_branch2a_relu);

    SharedPtr<convolution2d::Batch<float> > res5b_branch2b(new convolution2d::Batch<float>());
    res5b_branch2b->parameter.nKernels           = 512;
    res5b_branch2b->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    res5b_branch2b->parameter.strides            = convolution2d::Strides(1, 1);
    res5b_branch2b->parameter.paddings           = convolution2d::Paddings(1, 1);
    res5b_branch2b->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res5b_branch2b->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res5b_branch2b_id               = topology->add(res5b_branch2b);

    SharedPtr<batch_normalization::Batch<float> > bn5b_branch2b(new batch_normalization::Batch<float>());
    const size_t bn5b_branch2b_id = topology->add(bn5b_branch2b);

    SharedPtr<relu::Batch<float> > res5b_branch2b_relu(new relu::Batch<float>());
    const size_t res5b_branch2b_relu_id = topology->add(res5b_branch2b_relu);

    SharedPtr<convolution2d::Batch<float> > res5b_branch2c(new convolution2d::Batch<float>());
    res5b_branch2c->parameter.nKernels           = 2048;
    res5b_branch2c->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res5b_branch2c->parameter.strides            = convolution2d::Strides(1, 1);
    res5b_branch2c->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res5b_branch2c->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res5b_branch2c_id               = topology->add(res5b_branch2c);

    SharedPtr<batch_normalization::Batch<float> > bn5b_branch2c(new batch_normalization::Batch<float>());
    const size_t bn5b_branch2c_id = topology->add(bn5b_branch2c);

    SharedPtr<eltwise_sum::Batch<float> > res5b(new eltwise_sum::Batch<float>());
    const size_t res5b_id = topology->add(res5b);

    SharedPtr<relu::Batch<float> > res5b_relu(new relu::Batch<float>());
    const size_t res5b_relu_id = topology->add(res5b_relu);

    SharedPtr<split::Batch<float> > res5b_relu_split16(new split::Batch<float>(2, 2));
    const size_t res5b_relu_split16_id = topology->add(res5b_relu_split16);

    SharedPtr<convolution2d::Batch<float> > res5c_branch2a(new convolution2d::Batch<float>());
    res5c_branch2a->parameter.nKernels           = 512;
    res5c_branch2a->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res5c_branch2a->parameter.strides            = convolution2d::Strides(1, 1);
    res5c_branch2a->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res5c_branch2a->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res5c_branch2a_id               = topology->add(res5c_branch2a);

    SharedPtr<batch_normalization::Batch<float> > bn5c_branch2a(new batch_normalization::Batch<float>());
    const size_t bn5c_branch2a_id = topology->add(bn5c_branch2a);

    SharedPtr<relu::Batch<float> > res5c_branch2a_relu(new relu::Batch<float>());
    const size_t res5c_branch2a_relu_id = topology->add(res5c_branch2a_relu);

    SharedPtr<convolution2d::Batch<float> > res5c_branch2b(new convolution2d::Batch<float>());
    res5c_branch2b->parameter.nKernels           = 512;
    res5c_branch2b->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    res5c_branch2b->parameter.strides            = convolution2d::Strides(1, 1);
    res5c_branch2b->parameter.paddings           = convolution2d::Paddings(1, 1);
    res5c_branch2b->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res5c_branch2b->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res5c_branch2b_id               = topology->add(res5c_branch2b);

    SharedPtr<batch_normalization::Batch<float> > bn5c_branch2b(new batch_normalization::Batch<float>());
    const size_t bn5c_branch2b_id = topology->add(bn5c_branch2b);

    SharedPtr<relu::Batch<float> > res5c_branch2b_relu(new relu::Batch<float>());
    const size_t res5c_branch2b_relu_id = topology->add(res5c_branch2b_relu);

    SharedPtr<convolution2d::Batch<float> > res5c_branch2c(new convolution2d::Batch<float>());
    res5c_branch2c->parameter.nKernels           = 2048;
    res5c_branch2c->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    res5c_branch2c->parameter.strides            = convolution2d::Strides(1, 1);
    res5c_branch2c->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    res5c_branch2c->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t res5c_branch2c_id               = topology->add(res5c_branch2c);

    SharedPtr<batch_normalization::Batch<float> > bn5c_branch2c(new batch_normalization::Batch<float>());
    const size_t bn5c_branch2c_id = topology->add(bn5c_branch2c);

    SharedPtr<eltwise_sum::Batch<float> > res5c(new eltwise_sum::Batch<float>());
    const size_t res5c_id = topology->add(res5c);

    SharedPtr<relu::Batch<float> > res5c_relu(new relu::Batch<float>());
    const size_t res5c_relu_id = topology->add(res5c_relu);

    SharedPtr<average_pooling2d::Batch<float> > pool5(new average_pooling2d::Batch<float>(4));
    pool5->parameter.kernelSizes = pooling2d::KernelSizes(7, 7);
    pool5->parameter.strides     = pooling2d::Strides(1, 1);
    const size_t pool5_id        = topology->add(pool5);

    SharedPtr<fullyconnected::Batch<float> > fc1000(new fullyconnected::Batch<float>(1000));
    fc1000->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    fc1000->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));
    const size_t fc1000_id               = topology->add(fc1000);

    SharedPtr<loss::softmax_cross::Batch<float> > loss_layer(new loss::softmax_cross::Batch<float>());
    const size_t loss_layer_id = topology->add(loss_layer);

    topology->get( conv1_id               ).addNext( bn_conv1_id            );
    topology->get( bn_conv1_id            ).addNext( conv1_relu_id          );
    topology->get( conv1_relu_id          ).addNext( pool1_id               );
    topology->get( pool1_id               ).addNext( pool1_split1_id        );
    topology->get( pool1_split1_id        ).addNext( res2a_branch2a_id      );
    topology->get( pool1_split1_id        ).addNext( res2a_branch1_id       );
    topology->get( res2a_branch1_id       ).addNext( bn2a_branch1_id        );
    topology->get( bn2a_branch1_id        ).addNext( res2a_id               );
    topology->get( res2a_branch2a_id      ).addNext( bn2a_branch2a_id       );
    topology->get( bn2a_branch2a_id       ).addNext( res2a_branch2a_relu_id );
    topology->get( res2a_branch2a_relu_id ).addNext( res2a_branch2b_id      );
    topology->get( res2a_branch2b_id      ).addNext( bn2a_branch2b_id       );
    topology->get( bn2a_branch2b_id       ).addNext( res2a_branch2b_relu_id );
    topology->get( res2a_branch2b_relu_id ).addNext( res2a_branch2c_id      );
    topology->get( res2a_branch2c_id      ).addNext( bn2a_branch2c_id       );
    topology->get( bn2a_branch2c_id       ).addNext( res2a_id               );
    topology->get( res2a_id               ).addNext( res2a_relu_id          );
    topology->get( res2a_relu_id          ).addNext( res2a_relu_split2_id   );
    topology->get( res2a_relu_split2_id   ).addNext( res2b_branch2a_id      );
    topology->get( res2a_relu_split2_id   ).addNext( res2b_id               );
    topology->get( res2b_branch2a_id      ).addNext( bn2b_branch2a_id       );
    topology->get( bn2b_branch2a_id       ).addNext( res2b_branch2a_relu_id );
    topology->get( res2b_branch2a_relu_id ).addNext( res2b_branch2b_id      );
    topology->get( res2b_branch2b_id      ).addNext( bn2b_branch2b_id       );
    topology->get( bn2b_branch2b_id       ).addNext( res2b_branch2b_relu_id );
    topology->get( res2b_branch2b_relu_id ).addNext( res2b_branch2c_id      );
    topology->get( res2b_branch2c_id      ).addNext( bn2b_branch2c_id       );
    topology->get( bn2b_branch2c_id       ).addNext( res2b_id               );
    topology->get( res2b_id               ).addNext( res2b_relu_id          );
    topology->get( res2b_relu_id          ).addNext( res2b_relu_split3_id   );
    topology->get( res2b_relu_split3_id   ).addNext( res2c_id               );
    topology->get( res2b_relu_split3_id   ).addNext( res2c_branch2a_id      );
    topology->get( res2c_branch2a_id      ).addNext( bn2c_branch2a_id       );
    topology->get( bn2c_branch2a_id       ).addNext( res2c_branch2a_relu_id );
    topology->get( res2c_branch2a_relu_id ).addNext( res2c_branch2b_id      );
    topology->get( res2c_branch2b_id      ).addNext( bn2c_branch2b_id       );
    topology->get( bn2c_branch2b_id       ).addNext( res2c_branch2b_relu_id );
    topology->get( res2c_branch2b_relu_id ).addNext( res2c_branch2c_id      );
    topology->get( res2c_branch2c_id      ).addNext( bn2c_branch2c_id       );
    topology->get( bn2c_branch2c_id       ).addNext( res2c_id               );
    topology->get( res2c_id               ).addNext( res2c_relu_id          );
    topology->get( res2c_relu_id          ).addNext( res2c_relu_split4_id   );
    topology->get( res2c_relu_split4_id   ).addNext( res3a_branch1_id       );
    topology->get( res2c_relu_split4_id   ).addNext( res3a_branch2a_id      );
    topology->get( res3a_branch1_id       ).addNext( bn3a_branch1_id        );
    topology->get( bn3a_branch1_id        ).addNext( res3a_id               );
    topology->get( res3a_branch2a_id      ).addNext( bn3a_branch2a_id       );
    topology->get( bn3a_branch2a_id       ).addNext( res3a_branch2a_relu_id );
    topology->get( res3a_branch2a_relu_id ).addNext( res3a_branch2b_id      );
    topology->get( res3a_branch2b_id      ).addNext( bn3a_branch2b_id       );
    topology->get( bn3a_branch2b_id       ).addNext( res3a_branch2b_relu_id );
    topology->get( res3a_branch2b_relu_id ).addNext( res3a_branch2c_id      );
    topology->get( res3a_branch2c_id      ).addNext( bn3a_branch2c_id       );
    topology->get( bn3a_branch2c_id       ).addNext( res3a_id               );
    topology->get( res3a_id               ).addNext( res3a_relu_id          );
    topology->get( res3a_relu_id          ).addNext( res3a_relu_split5_id   );
    topology->get( res3a_relu_split5_id   ).addNext( res3b_id               );
    topology->get( res3a_relu_split5_id   ).addNext( res3b_branch2a_id      );
    topology->get( res3b_branch2a_id      ).addNext( bn3b_branch2a_id       );
    topology->get( bn3b_branch2a_id       ).addNext( res3b_branch2a_relu_id );
    topology->get( res3b_branch2a_relu_id ).addNext( res3b_branch2b_id      );
    topology->get( res3b_branch2b_id      ).addNext( bn3b_branch2b_id       );
    topology->get( bn3b_branch2b_id       ).addNext( res3b_branch2b_relu_id );
    topology->get( res3b_branch2b_relu_id ).addNext( res3b_branch2c_id      );
    topology->get( res3b_branch2c_id      ).addNext( bn3b_branch2c_id       );
    topology->get( bn3b_branch2c_id       ).addNext( res3b_id               );
    topology->get( res3b_id               ).addNext( res3b_relu_id          );
    topology->get( res3b_relu_id          ).addNext( res3b_relu_split6_id   );
    topology->get( res3b_relu_split6_id   ).addNext( res3c_id               );
    topology->get( res3b_relu_split6_id   ).addNext( res3c_branch2a_id      );
    topology->get( res3c_branch2a_id      ).addNext( bn3c_branch2a_id       );
    topology->get( bn3c_branch2a_id       ).addNext( res3c_branch2a_relu_id );
    topology->get( res3c_branch2a_relu_id ).addNext( res3c_branch2b_id      );
    topology->get( res3c_branch2b_id      ).addNext( bn3c_branch2b_id       );
    topology->get( bn3c_branch2b_id       ).addNext( res3c_branch2b_relu_id );
    topology->get( res3c_branch2b_relu_id ).addNext( res3c_branch2c_id      );
    topology->get( res3c_branch2c_id      ).addNext( bn3c_branch2c_id       );
    topology->get( bn3c_branch2c_id       ).addNext( res3c_id               );
    topology->get( res3c_id               ).addNext( res3c_relu_id          );
    topology->get( res3c_relu_id          ).addNext( res3c_relu_split7_id   );
    topology->get( res3c_relu_split7_id   ).addNext( res3d_branch2a_id      );
    topology->get( res3c_relu_split7_id   ).addNext( res3d_id               );
    topology->get( res3d_branch2a_id      ).addNext( bn3d_branch2a_id       );
    topology->get( bn3d_branch2a_id       ).addNext( res3d_branch2a_relu_id );
    topology->get( res3d_branch2a_relu_id ).addNext( res3d_branch2b_id      );
    topology->get( res3d_branch2b_id      ).addNext( bn3d_branch2b_id       );
    topology->get( bn3d_branch2b_id       ).addNext( res3d_branch2b_relu_id );
    topology->get( res3d_branch2b_relu_id ).addNext( res3d_branch2c_id      );
    topology->get( res3d_branch2c_id      ).addNext( bn3d_branch2c_id       );
    topology->get( bn3d_branch2c_id       ).addNext( res3d_id               );
    topology->get( res3d_id               ).addNext( res3d_relu_id          );
    topology->get( res3d_relu_id          ).addNext( res3d_relu_split8_id   );
    topology->get( res3d_relu_split8_id   ).addNext( res4a_branch1_id       );
    topology->get( res3d_relu_split8_id   ).addNext( res4a_branch2a_id      );
    topology->get( res4a_branch1_id       ).addNext( bn4a_branch1_id        );
    topology->get( bn4a_branch1_id        ).addNext( res4a_id               );
    topology->get( res4a_branch2a_id      ).addNext( bn4a_branch2a_id       );
    topology->get( bn4a_branch2a_id       ).addNext( res4a_branch2a_relu_id );
    topology->get( res4a_branch2a_relu_id ).addNext( res4a_branch2b_id      );
    topology->get( res4a_branch2b_id      ).addNext( bn4a_branch2b_id       );
    topology->get( bn4a_branch2b_id       ).addNext( res4a_branch2b_relu_id );
    topology->get( res4a_branch2b_relu_id ).addNext( res4a_branch2c_id      );
    topology->get( res4a_branch2c_id      ).addNext( bn4a_branch2c_id       );
    topology->get( bn4a_branch2c_id       ).addNext( res4a_id               );
    topology->get( res4a_id               ).addNext( res4a_relu_id          );
    topology->get( res4a_relu_id          ).addNext( res4a_relu_split9_id   );
    topology->get( res4a_relu_split9_id   ).addNext( res4b_id               );
    topology->get( res4a_relu_split9_id   ).addNext( res4b_branch2a_id      );
    topology->get( res4b_branch2a_id      ).addNext( bn4b_branch2a_id       );
    topology->get( bn4b_branch2a_id       ).addNext( res4b_branch2a_relu_id );
    topology->get( res4b_branch2a_relu_id ).addNext( res4b_branch2b_id      );
    topology->get( res4b_branch2b_id      ).addNext( bn4b_branch2b_id       );
    topology->get( bn4b_branch2b_id       ).addNext( res4b_branch2b_relu_id );
    topology->get( res4b_branch2b_relu_id ).addNext( res4b_branch2c_id      );
    topology->get( res4b_branch2c_id      ).addNext( bn4b_branch2c_id       );
    topology->get( bn4b_branch2c_id       ).addNext( res4b_id               );
    topology->get( res4b_id               ).addNext( res4b_relu_id          );
    topology->get( res4b_relu_id          ).addNext( res4b_relu_split10_id  );
    topology->get( res4b_relu_split10_id  ).addNext( res4c_branch2a_id      );
    topology->get( res4b_relu_split10_id  ).addNext( res4c_id               );
    topology->get( res4c_branch2a_id      ).addNext( bn4c_branch2a_id       );
    topology->get( bn4c_branch2a_id       ).addNext( res4c_branch2a_relu_id );
    topology->get( res4c_branch2a_relu_id ).addNext( res4c_branch2b_id      );
    topology->get( res4c_branch2b_id      ).addNext( bn4c_branch2b_id       );
    topology->get( bn4c_branch2b_id       ).addNext( res4c_branch2b_relu_id );
    topology->get( res4c_branch2b_relu_id ).addNext( res4c_branch2c_id      );
    topology->get( res4c_branch2c_id      ).addNext( bn4c_branch2c_id       );
    topology->get( bn4c_branch2c_id       ).addNext( res4c_id               );
    topology->get( res4c_id               ).addNext( res4c_relu_id          );
    topology->get( res4c_relu_id          ).addNext( res4c_relu_split11_id  );
    topology->get( res4c_relu_split11_id  ).addNext( res4d_id               );
    topology->get( res4c_relu_split11_id  ).addNext( res4d_branch2a_id      );
    topology->get( res4d_branch2a_id      ).addNext( bn4d_branch2a_id       );
    topology->get( bn4d_branch2a_id       ).addNext( res4d_branch2a_relu_id );
    topology->get( res4d_branch2a_relu_id ).addNext( res4d_branch2b_id      );
    topology->get( res4d_branch2b_id      ).addNext( bn4d_branch2b_id       );
    topology->get( bn4d_branch2b_id       ).addNext( res4d_branch2b_relu_id );
    topology->get( res4d_branch2b_relu_id ).addNext( res4d_branch2c_id      );
    topology->get( res4d_branch2c_id      ).addNext( bn4d_branch2c_id       );
    topology->get( bn4d_branch2c_id       ).addNext( res4d_id               );
    topology->get( res4d_id               ).addNext( res4d_relu_id          );
    topology->get( res4d_relu_id          ).addNext( res4d_relu_split12_id  );
    topology->get( res4d_relu_split12_id  ).addNext( res4e_branch2a_id      );
    topology->get( res4d_relu_split12_id  ).addNext( res4e_id               );
    topology->get( res4e_branch2a_id      ).addNext( bn4e_branch2a_id       );
    topology->get( bn4e_branch2a_id       ).addNext( res4e_branch2a_relu_id );
    topology->get( res4e_branch2a_relu_id ).addNext( res4e_branch2b_id      );
    topology->get( res4e_branch2b_id      ).addNext( bn4e_branch2b_id       );
    topology->get( bn4e_branch2b_id       ).addNext( res4e_branch2b_relu_id );
    topology->get( res4e_branch2b_relu_id ).addNext( res4e_branch2c_id      );
    topology->get( res4e_branch2c_id      ).addNext( bn4e_branch2c_id       );
    topology->get( bn4e_branch2c_id       ).addNext( res4e_id               );
    topology->get( res4e_id               ).addNext( res4e_relu_id          );
    topology->get( res4e_relu_id          ).addNext( res4e_relu_split13_id  );
    topology->get( res4e_relu_split13_id  ).addNext( res4f_id               );
    topology->get( res4e_relu_split13_id  ).addNext( res4f_branch2a_id      );
    topology->get( res4f_branch2a_id      ).addNext( bn4f_branch2a_id       );
    topology->get( bn4f_branch2a_id       ).addNext( res4f_branch2a_relu_id );
    topology->get( res4f_branch2a_relu_id ).addNext( res4f_branch2b_id      );
    topology->get( res4f_branch2b_id      ).addNext( bn4f_branch2b_id       );
    topology->get( bn4f_branch2b_id       ).addNext( res4f_branch2b_relu_id );
    topology->get( res4f_branch2b_relu_id ).addNext( res4f_branch2c_id      );
    topology->get( res4f_branch2c_id      ).addNext( bn4f_branch2c_id       );
    topology->get( bn4f_branch2c_id       ).addNext( res4f_id               );
    topology->get( res4f_id               ).addNext( res4f_relu_id          );
    topology->get( res4f_relu_id          ).addNext( res4f_relu_split14_id  );
    topology->get( res4f_relu_split14_id  ).addNext( res5a_branch1_id       );
    topology->get( res4f_relu_split14_id  ).addNext( res5a_branch2a_id      );
    topology->get( res5a_branch1_id       ).addNext( bn5a_branch1_id        );
    topology->get( bn5a_branch1_id        ).addNext( res5a_id               );
    topology->get( res5a_branch2a_id      ).addNext( bn5a_branch2a_id       );
    topology->get( bn5a_branch2a_id       ).addNext( res5a_branch2a_relu_id );
    topology->get( res5a_branch2a_relu_id ).addNext( res5a_branch2b_id      );
    topology->get( res5a_branch2b_id      ).addNext( bn5a_branch2b_id       );
    topology->get( bn5a_branch2b_id       ).addNext( res5a_branch2b_relu_id );
    topology->get( res5a_branch2b_relu_id ).addNext( res5a_branch2c_id      );
    topology->get( res5a_branch2c_id      ).addNext( bn5a_branch2c_id       );
    topology->get( bn5a_branch2c_id       ).addNext( res5a_id               );
    topology->get( res5a_id               ).addNext( res5a_relu_id          );
    topology->get( res5a_relu_id          ).addNext( res5a_relu_split15_id  );
    topology->get( res5a_relu_split15_id  ).addNext( res5b_branch2a_id      );
    topology->get( res5a_relu_split15_id  ).addNext( res5b_id               );
    topology->get( res5b_branch2a_id      ).addNext( bn5b_branch2a_id       );
    topology->get( bn5b_branch2a_id       ).addNext( res5b_branch2a_relu_id );
    topology->get( res5b_branch2a_relu_id ).addNext( res5b_branch2b_id      );
    topology->get( res5b_branch2b_id      ).addNext( bn5b_branch2b_id       );
    topology->get( bn5b_branch2b_id       ).addNext( res5b_branch2b_relu_id );
    topology->get( res5b_branch2b_relu_id ).addNext( res5b_branch2c_id      );
    topology->get( res5b_branch2c_id      ).addNext( bn5b_branch2c_id       );
    topology->get( bn5b_branch2c_id       ).addNext( res5b_id               );
    topology->get( res5b_id               ).addNext( res5b_relu_id          );
    topology->get( res5b_relu_id          ).addNext( res5b_relu_split16_id  );
    topology->get( res5b_relu_split16_id  ).addNext( res5c_branch2a_id      );
    topology->get( res5b_relu_split16_id  ).addNext( res5c_id               );
    topology->get( res5c_branch2a_id      ).addNext( bn5c_branch2a_id       );
    topology->get( bn5c_branch2a_id       ).addNext( res5c_branch2a_relu_id );
    topology->get( res5c_branch2a_relu_id ).addNext( res5c_branch2b_id      );
    topology->get( res5c_branch2b_id      ).addNext( bn5c_branch2b_id       );
    topology->get( bn5c_branch2b_id       ).addNext( res5c_branch2b_relu_id );
    topology->get( res5c_branch2b_relu_id ).addNext( res5c_branch2c_id      );
    topology->get( res5c_branch2c_id      ).addNext( bn5c_branch2c_id       );
    topology->get( bn5c_branch2c_id       ).addNext( res5c_id               );
    topology->get( res5c_id               ).addNext( res5c_relu_id          );
    topology->get( res5c_relu_id          ).addNext( pool5_id               );
    topology->get( pool5_id               ).addNext( fc1000_id              );
    topology->get( fc1000_id              ).addNext( loss_layer_id          );

    return topology;
}

#endif
