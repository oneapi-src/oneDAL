/* file: daal_alexnet.h */
/*******************************************************************************
* Copyright 2017-2019 Intel Corporation.
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

#ifndef _DAAL_ALEXNET_H
#define _DAAL_ALEXNET_H

#include "daal_defines.h"

training::TopologyPtr configureNet()
{
    /* convolution: 11x11@96 + 4x4s */
    SharedPtr<convolution2d::Batch<> > convolution1(new convolution2d::Batch<>() );
    convolution1->parameter.kernelSizes        = convolution2d::KernelSizes(11, 11);
    convolution1->parameter.strides            = convolution2d::Strides(4, 4);
    convolution1->parameter.nKernels           = 96;
    convolution1->parameter.weightsInitializer = GaussianInitializerPtr(new GaussianInitializer(0, 0.01));
    convolution1->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));

    /* relu */
    SharedPtr<relu::Batch<> > relu1(new relu::Batch<>);

    /* lrn: alpha=0.0001, beta=0.75, local_size=5 */
    SharedPtr<lrn::Batch<> > lrn1(new lrn::Batch<>());
    lrn1->parameter.kappa   = 1;
    lrn1->parameter.nAdjust = 5;
    lrn1->parameter.alpha   = 0.0001 / lrn1->parameter.nAdjust;
    lrn1->parameter.beta    = 0.75;

    /* pooling: 3x3 + 2x2s */
    SharedPtr<maximum_pooling2d::Batch<> > maxpooling1(new maximum_pooling2d::Batch<>(4));
    maxpooling1->parameter.kernelSizes = pooling2d::KernelSizes(3, 3);
    maxpooling1->parameter.paddings    = pooling2d::Paddings(0, 0);
    maxpooling1->parameter.strides     = pooling2d::Strides(2, 2);

    /* convolution: 5x5@256 + 1x1s */
    SharedPtr<convolution2d::Batch<> > convolution2(new convolution2d::Batch<>());
    convolution2->parameter.kernelSizes        = convolution2d::KernelSizes(5, 5);
    convolution2->parameter.strides            = convolution2d::Strides(1, 1);
    convolution2->parameter.paddings           = convolution2d::Paddings(2, 2);
    convolution2->parameter.nKernels           = 256;
    convolution2->parameter.nGroups            = 2;
    convolution2->parameter.weightsInitializer = GaussianInitializerPtr(new GaussianInitializer(0, 0.01));
    convolution2->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));

    /* relu */
    SharedPtr<relu::Batch<> > relu2(new relu::Batch<>);

    /* lrn: alpha=0.0001, beta=0.75, local_size=5 */
    SharedPtr<lrn::Batch<> > lrn2(new lrn::Batch<>());
    lrn2->parameter.kappa   = 1;
    lrn2->parameter.nAdjust = 5;
    lrn2->parameter.alpha   = 0.0001 / lrn2->parameter.nAdjust;
    lrn2->parameter.beta    = 0.75;

    /* pooling: 3x3 + 2x2s */
    SharedPtr<maximum_pooling2d::Batch<> > maxpooling2(new maximum_pooling2d::Batch<>(4));
    maxpooling2->parameter.kernelSizes = pooling2d::KernelSizes(3, 3);
    maxpooling2->parameter.paddings    = pooling2d::Paddings(0, 0);
    maxpooling2->parameter.strides     = pooling2d::Strides(2, 2);

    /* convolution: 3x3@384 + 2x2s */
    SharedPtr<convolution2d::Batch<> > convolution3(new convolution2d::Batch<>());
    convolution3->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    convolution3->parameter.paddings           = convolution2d::Paddings(1, 1);
    convolution3->parameter.strides            = convolution2d::Strides(1, 1);
    convolution3->parameter.nKernels           = 384;
    convolution3->parameter.weightsInitializer = GaussianInitializerPtr(new GaussianInitializer(0, 0.01));
    convolution3->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));

    /* relu */
    SharedPtr<relu::Batch<> > relu3(new relu::Batch<>);

    /* convolution: 3x3@384 + 2x2s */
    SharedPtr<convolution2d::Batch<> > convolution4(new convolution2d::Batch<>());
    convolution4->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    convolution4->parameter.paddings           = convolution2d::Paddings(1, 1);
    convolution4->parameter.strides            = convolution2d::Strides(1, 1);
    convolution4->parameter.nKernels           = 384;
    convolution4->parameter.nGroups            = 2;
    convolution4->parameter.weightsInitializer = GaussianInitializerPtr(new GaussianInitializer(0, 0.01));
    convolution4->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));

    /* relu */
    SharedPtr<relu::Batch<> > relu4(new relu::Batch<>);

    /* convolution: 3x3@256 + 2x2s */
    SharedPtr<convolution2d::Batch<> > convolution5(new convolution2d::Batch<>());
    convolution5->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    convolution5->parameter.paddings           = convolution2d::Paddings(1, 1);
    convolution5->parameter.strides            = convolution2d::Strides(1, 1);
    convolution5->parameter.nKernels           = 256;
    convolution5->parameter.nGroups            = 2;
    convolution5->parameter.weightsInitializer = GaussianInitializerPtr(new GaussianInitializer(0, 0.01));
    convolution5->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));

    /* relu */
    SharedPtr<relu::Batch<> > relu5(new relu::Batch<>);

    /* pooling: 3x3 + 2x2s */
    SharedPtr<maximum_pooling2d::Batch<> > maxpooling5(new maximum_pooling2d::Batch<>(4));
    maxpooling5->parameter.kernelSizes = pooling2d::KernelSizes(3, 3);
    maxpooling5->parameter.paddings    = pooling2d::Paddings(0, 0);
    maxpooling5->parameter.strides     = pooling2d::Strides(2, 2);

    /* fullyconnected: n = 4096 */
    SharedPtr<fullyconnected::Batch<> > fullyconnected6(new fullyconnected::Batch<>(4096));
    fullyconnected6->parameter.weightsInitializer = GaussianInitializerPtr(new GaussianInitializer(0, 0.01));
    fullyconnected6->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));

    /* relu */
    SharedPtr<relu::Batch<> > relu6(new relu::Batch<>);

    /* dropout: p = 0.5 */
    SharedPtr<dropout::Batch<> > dropout6(new dropout::Batch<>());
    dropout6->parameter.retainRatio = 0.5;

    /* fullyconnected: n = 4096 */
    SharedPtr<fullyconnected::Batch<> > fullyconnected7(new fullyconnected::Batch<>(4096));
    fullyconnected7->parameter.weightsInitializer = GaussianInitializerPtr(new GaussianInitializer(0, 0.005));
    fullyconnected7->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));

    /* relu */
    SharedPtr<relu::Batch<> > relu7(new relu::Batch<>);

    /* dropout: p = 0.5 */
    SharedPtr<dropout::Batch<> > dropout7(new dropout::Batch<>());
    dropout7->parameter.retainRatio = 0.5;

    /* fullyconnected: n = 1000 */
    SharedPtr<fullyconnected::Batch<> > fullyconnected8(new fullyconnected::Batch<>(1000));
    fullyconnected8->parameter.weightsInitializer = GaussianInitializerPtr(new GaussianInitializer(0, 0.01));
    fullyconnected8->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0, 0));

    /* softmax + crossentropy loss */
    SharedPtr<loss::softmax_cross::Batch<> > softmax(new loss::softmax_cross::Batch<>());

    /* Create AlexNet Topology */
    training::TopologyPtr topology(new training::Topology());
    const size_t conv1 = topology->add( convolution1);
    const size_t r1    = topology->add( relu1           ); topology->get( conv1 ).addNext(r1);
    const size_t l1    = topology->add( lrn1            ); topology->get( r1    ).addNext(l1);
    const size_t pool1 = topology->add( maxpooling1     ); topology->get( l1    ).addNext(pool1);
    const size_t conv2 = topology->add( convolution2    ); topology->get( pool1 ).addNext(conv2);
    const size_t r2    = topology->add( relu2           ); topology->get( conv2 ).addNext(r2);
    const size_t l2    = topology->add( lrn2            ); topology->get( r2    ).addNext(l2);
    const size_t pool2 = topology->add( maxpooling2     ); topology->get( l2    ).addNext(pool2);
    const size_t conv3 = topology->add( convolution3    ); topology->get( pool2 ).addNext(conv3);
    const size_t r3    = topology->add( relu3           ); topology->get( conv3 ).addNext(r3);
    const size_t conv4 = topology->add( convolution4    ); topology->get( r3    ).addNext(conv4);
    const size_t r4    = topology->add( relu4           ); topology->get( conv4 ).addNext(r4);
    const size_t conv5 = topology->add( convolution5    ); topology->get( r4    ).addNext(conv5);
    const size_t r5    = topology->add( relu5           ); topology->get( conv5 ).addNext(r5);
    const size_t pool5 = topology->add( maxpooling5     ); topology->get( r5    ).addNext(pool5);
    const size_t fc6   = topology->add( fullyconnected6 ); topology->get( pool5 ).addNext(fc6);
    const size_t r6    = topology->add( relu6           ); topology->get( fc6   ).addNext(r6);
    const size_t drop6 = topology->add( dropout6        ); topology->get( r6    ).addNext(drop6);
    const size_t fc7   = topology->add( fullyconnected7 ); topology->get( drop6 ).addNext(fc7);
    const size_t r7    = topology->add( relu7           ); topology->get( fc7   ).addNext(r7);
    const size_t drop7 = topology->add( dropout7        ); topology->get( r7    ).addNext(drop7);
    const size_t fc8   = topology->add( fullyconnected8 ); topology->get( drop7 ).addNext(fc8);
    const size_t sm    = topology->add( softmax         ); topology->get( fc8   ).addNext(sm);
    return topology;
}

#endif
