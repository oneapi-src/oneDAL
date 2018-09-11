/* file: daal_googlenet_v1.h */
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

#ifndef _DAAL_GOOGLENET_V1_H
#define _DAAL_GOOGLENET_V1_H

#include "daal_defines.h"

training::TopologyPtr configureLossLayers()
{
    training::TopologyPtr topology(new training::Topology());

    /* pooling(loss1/ave_pool): 5x5 + 3x3s */
    SharedPtr<average_pooling2d::Batch<> > loss_ave_pool(new average_pooling2d::Batch<>(4));
    loss_ave_pool->parameter.kernelSizes = pooling2d::KernelSizes(5, 5);
    loss_ave_pool->parameter.strides     = pooling2d::Strides(3, 3);
    loss_ave_pool->parameter.paddings    = pooling2d::Paddings(0, 0);
    const size_t loss_ave_pool_id = topology->add(loss_ave_pool);

    /* convolution(loss/conv): 1x1@128 + 1x1s */
    SharedPtr<convolution2d::Batch<> > loss_conv(new convolution2d::Batch<>());
    loss_conv->parameter.nKernels           = 128;
    loss_conv->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    loss_conv->parameter.strides            = convolution2d::Strides(1, 1);
    loss_conv->parameter.paddings           = convolution2d::Paddings(0, 0);
    loss_conv->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    loss_conv->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0.2, 0.2));
    const size_t loss_conv_id = topology->add(loss_conv);

    /* relu(loss/relu_conv) */
    SharedPtr<relu::Batch<> > loss_relu_conv(new relu::Batch<>());
    const size_t loss_relu_conv_id = topology->add(loss_relu_conv);

    /* fullyconnected(loss/fc): n = 1024 */
    SharedPtr<fullyconnected::Batch<> > loss_fc(new fullyconnected::Batch<>(1024));
    loss_fc->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    loss_fc->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0.2, 0.2));
    const size_t loss_fc_id = topology->add(loss_fc);

    /* relu(loss/relu_fc) */
    SharedPtr<relu::Batch<> > loss_relu_fc(new relu::Batch<>());
    const size_t loss_relu_fc_id = topology->add(loss_relu_fc);

    /* dropout(loss/drop_fc): p = 0.7 */
    SharedPtr<dropout::Batch<> > loss_drop_fc(new dropout::Batch<>());
    loss_drop_fc->parameter.retainRatio = 0.7;
    const size_t loss_drop_fc_id = topology->add(loss_drop_fc);

    /* fullyconnected(loss/classifier): n = 1000 */
    SharedPtr<fullyconnected::Batch<> > loss_classifier(new fullyconnected::Batch<>(1000));
    loss_classifier->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    loss_classifier->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0.0, 0.0));
    const size_t loss_classifier_id = topology->add(loss_classifier);

    /* softmax + crossentropy loss(loss/loss) */
    SharedPtr<loss::softmax_cross::Batch<> > loss_loss(new loss::softmax_cross::Batch<>());
    const size_t loss_loss_id = topology->add(loss_loss);

    topology->get( loss_ave_pool_id   ).addNext( loss_conv_id       );
    topology->get( loss_conv_id       ).addNext( loss_relu_conv_id  );
    topology->get( loss_relu_conv_id  ).addNext( loss_fc_id         );
    topology->get( loss_fc_id         ).addNext( loss_relu_fc_id    );
    topology->get( loss_relu_fc_id    ).addNext( loss_drop_fc_id    );
    topology->get( loss_drop_fc_id    ).addNext( loss_classifier_id );
    topology->get( loss_classifier_id ).addNext( loss_loss_id       );

    return topology;
}

training::TopologyPtr configureInception(size_t nKernels1, size_t nKernels2, size_t nKernels3,
                                         size_t nKernels4, size_t nKernels5, size_t nKernels6, bool hasLoss=false)
{
    training::TopologyPtr topology(new training::Topology());

    size_t splitSuccessors = 4;
    if (hasLoss) { splitSuccessors++; }

    /* split */
    SharedPtr<split::Batch<> > split(new split::Batch<>(splitSuccessors, splitSuccessors));
    const size_t split_id = topology->add(split);

    /* convolution(inception/1x1): 1x1@nKernels1 + 1x1s*/
    SharedPtr<convolution2d::Batch<> > inception_1x1(new convolution2d::Batch<>());
    inception_1x1->parameter.nKernels           = nKernels1;
    inception_1x1->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    inception_1x1->parameter.strides            = convolution2d::Strides(1, 1);
    inception_1x1->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    inception_1x1->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0.2, 0.2));
    const size_t inception_1x1_id = topology->add(inception_1x1);

    /* relu(inception/relu_1x1) */
    SharedPtr<relu::Batch<> > inception_relu_1x1(new relu::Batch<>());
    const size_t inception_relu_1x1_id = topology->add(inception_relu_1x1);

    /* convolution(inception/3x3_reduce): 1x1@nKernels2 + 1x1s */
    SharedPtr<convolution2d::Batch<> > inception_3x3_reduce(new convolution2d::Batch<>());
    inception_3x3_reduce->parameter.nKernels           = nKernels2;
    inception_3x3_reduce->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    inception_3x3_reduce->parameter.strides            = convolution2d::Strides(1, 1);
    inception_3x3_reduce->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    inception_3x3_reduce->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0.2, 0.2));
    const size_t inception_3x3_reduce_id = topology->add(inception_3x3_reduce);

    /* relu(inception/relu_3x3_reduce) */
    SharedPtr<relu::Batch<> > inception_relu_3x3_reduce(new relu::Batch<>());
    const size_t inception_relu_3x3_reduce_id = topology->add(inception_relu_3x3_reduce);

    /* convolution(inception/3x3): 3x3@nKernels3 + 1x1s */
    SharedPtr<convolution2d::Batch<> > inception_3x3(new convolution2d::Batch<>());
    inception_3x3->parameter.nKernels           = nKernels3;
    inception_3x3->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    inception_3x3->parameter.strides            = convolution2d::Strides(1, 1);
    inception_3x3->parameter.paddings           = convolution2d::Paddings(1, 1);
    inception_3x3->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    inception_3x3->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0.2, 0.2));
    const size_t inception_3x3_id = topology->add(inception_3x3);

    /* relu(inception/relu_3x3) */
    SharedPtr<relu::Batch<> > inception_relu_3x3(new relu::Batch<>());
    const size_t inception_relu_3x3_id = topology->add(inception_relu_3x3);

    /* convolution(inception/5x5_reduce): 1x1@nKernels4 + 1x1s */
    SharedPtr<convolution2d::Batch<> > inception_5x5_reduce(new convolution2d::Batch<>());
    inception_5x5_reduce->parameter.nKernels           = nKernels4;
    inception_5x5_reduce->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    inception_5x5_reduce->parameter.strides            = convolution2d::Strides(1, 1);
    inception_5x5_reduce->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    inception_5x5_reduce->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0.2, 0.2));
    const size_t inception_5x5_reduce_id = topology->add(inception_5x5_reduce);

    /* relu(inception/relu_5x5_reduce) */
    SharedPtr<relu::Batch<> > inception_relu_5x5_reduce(new relu::Batch<>());
    const size_t inception_relu_5x5_reduce_id = topology->add(inception_relu_5x5_reduce);

    /* convolution(inception/5x5): 5x5@nKernels5 + 1x1s */
    SharedPtr<convolution2d::Batch<> > inception_5x5(new convolution2d::Batch<>());
    inception_5x5->parameter.nKernels           = nKernels5;
    inception_5x5->parameter.kernelSizes        = convolution2d::KernelSizes(5, 5);
    inception_5x5->parameter.strides            = convolution2d::Strides(1, 1);
    inception_5x5->parameter.paddings           = convolution2d::Paddings(2, 2);
    inception_5x5->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    inception_5x5->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0.2, 0.2));
    const size_t inception_5x5_id = topology->add(inception_5x5);

    /* relu(inception/relu_5x5) */
    SharedPtr<relu::Batch<> > inception_relu_5x5(new relu::Batch<>());
    const size_t inception_relu_5x5_id = topology->add(inception_relu_5x5);

    /* pooling(inception/pool): 3x3 + 1x1s */
    SharedPtr<maximum_pooling2d::Batch<> > inception_pool(new maximum_pooling2d::Batch<>(4));
    inception_pool->parameter.kernelSizes = pooling2d::KernelSizes(3, 3);
    inception_pool->parameter.strides     = pooling2d::Strides(1, 1);
    inception_pool->parameter.paddings    = pooling2d::Paddings(1, 1);
    const size_t inception_pool_id = topology->add(inception_pool);

    /* convolution(inception/pool_proj): 1x1@nKernels6 + 1x1s */
    SharedPtr<convolution2d::Batch<> > inception_pool_proj(new convolution2d::Batch<>());
    inception_pool_proj->parameter.nKernels           = nKernels6;
    inception_pool_proj->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    inception_pool_proj->parameter.strides            = convolution2d::Strides(1, 1);
    inception_pool_proj->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    inception_pool_proj->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0.2, 0.2));
    const size_t inception_pool_proj_id = topology->add(inception_pool_proj);

    /* relu(inception/relu_pool_proj) */
    SharedPtr<relu::Batch<> > inception_relu_pool_proj(new relu::Batch<>());
    const size_t inception_relu_pool_proj_id = topology->add(inception_relu_pool_proj);

    /* concat(inception/output) */
    SharedPtr<concat::Batch<> > inception_output(new concat::Batch<>());
    inception_output->parameter.concatDimension = 1;
    const size_t inception_output_id = topology->add(inception_output);

    topology->get( split_id                     ).addNext( inception_3x3_reduce_id      );
    topology->get( split_id                     ).addNext( inception_pool_id            );
    topology->get( split_id                     ).addNext( inception_1x1_id             );
    topology->get( split_id                     ).addNext( inception_5x5_reduce_id      );
    topology->get( inception_1x1_id             ).addNext( inception_relu_1x1_id        );
    topology->get( inception_relu_1x1_id        ).addNext( inception_output_id          );
    topology->get( inception_3x3_reduce_id      ).addNext( inception_relu_3x3_reduce_id );
    topology->get( inception_relu_3x3_reduce_id ).addNext( inception_3x3_id             );
    topology->get( inception_3x3_id             ).addNext( inception_relu_3x3_id        );
    topology->get( inception_relu_3x3_id        ).addNext( inception_output_id          );
    topology->get( inception_5x5_reduce_id      ).addNext( inception_relu_5x5_reduce_id );
    topology->get( inception_relu_5x5_reduce_id ).addNext( inception_5x5_id             );
    topology->get( inception_5x5_id             ).addNext( inception_relu_5x5_id        );
    topology->get( inception_relu_5x5_id        ).addNext( inception_output_id          );
    topology->get( inception_pool_id            ).addNext( inception_pool_proj_id       );
    topology->get( inception_pool_proj_id       ).addNext( inception_relu_pool_proj_id  );
    topology->get( inception_relu_pool_proj_id  ).addNext( inception_output_id          );

    return topology;
}

training::TopologyPtr configureNet()
{
    training::TopologyPtr topology(new training::Topology());

    /* convolution(conv1/7x7_s2): 7x7@64 + 2x2s */
    SharedPtr<convolution2d::Batch<> > conv1_7x7_s2(new convolution2d::Batch<>());
    conv1_7x7_s2->parameter.nKernels           = 64;
    conv1_7x7_s2->parameter.kernelSizes        = convolution2d::KernelSizes(7, 7);
    conv1_7x7_s2->parameter.strides            = convolution2d::Strides(2, 2);
    conv1_7x7_s2->parameter.paddings           = convolution2d::Paddings(3, 3);
    conv1_7x7_s2->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    conv1_7x7_s2->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0.2, 0.2));
    const size_t conv1_7x7_s2_id = topology->add(conv1_7x7_s2);

    /* relu(conv1/relu_7x7) */
    SharedPtr<relu::Batch<> > conv1_relu_7x7(new relu::Batch<>());
    const size_t conv1_relu_7x7_id = topology->add(conv1_relu_7x7);

    /* pooling(pool1/3x3_s2): 3x3 + 2x2s */
    SharedPtr<maximum_pooling2d::Batch<> > pool1_3x3_s2(new maximum_pooling2d::Batch<>(4));
    pool1_3x3_s2->parameter.kernelSizes = pooling2d::KernelSizes(3, 3);
    pool1_3x3_s2->parameter.strides     = pooling2d::Strides(2, 2);
    const size_t pool1_3x3_s2_id = topology->add(pool1_3x3_s2);

    /* lrn(pool1/norm1): alpha=0.0001, beta=0.75, local_size=5 */
    SharedPtr<lrn::Batch<> > pool1_norm1(new lrn::Batch<>());
    pool1_norm1->parameter.kappa   = 1;
    pool1_norm1->parameter.nAdjust = 5;
    pool1_norm1->parameter.beta    = 0.75;
    pool1_norm1->parameter.alpha   = 0.0001 / pool1_norm1->parameter.nAdjust;
    const size_t pool1_norm1_id = topology->add(pool1_norm1);

    /* convolution(conv2/3x3_reduce): 1x1@64 + 1x1s */
    SharedPtr<convolution2d::Batch<> > conv2_3x3_reduce(new convolution2d::Batch<>());
    conv2_3x3_reduce->parameter.nKernels           = 64;
    conv2_3x3_reduce->parameter.kernelSizes        = convolution2d::KernelSizes(1, 1);
    conv2_3x3_reduce->parameter.strides            = convolution2d::Strides(1, 1);
    conv2_3x3_reduce->parameter.paddings           = convolution2d::Paddings(0, 0);
    conv2_3x3_reduce->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    conv2_3x3_reduce->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0.2, 0.2));
    const size_t conv2_3x3_reduce_id = topology->add(conv2_3x3_reduce);

    /* convolution(conv2/relu_3x3_reduce) */
    SharedPtr<relu::Batch<> > conv2_relu_3x3_reduce(new relu::Batch<>());
    const size_t conv2_relu_3x3_reduce_id = topology->add(conv2_relu_3x3_reduce);

    /* convolution(conv2/3x3): 3x3@192 + 1x1s */
    SharedPtr<convolution2d::Batch<> > conv2_3x3(new convolution2d::Batch<>());
    conv2_3x3->parameter.nKernels           = 192;
    conv2_3x3->parameter.kernelSizes        = convolution2d::KernelSizes(3, 3);
    conv2_3x3->parameter.strides            = convolution2d::Strides(1, 1);
    conv2_3x3->parameter.paddings           = convolution2d::Paddings(1, 1);
    conv2_3x3->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    conv2_3x3->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0.2, 0.2));
    const size_t conv2_3x3_id = topology->add(conv2_3x3);

    /* relu(conv2/relu_3x3) */
    SharedPtr<relu::Batch<> > conv2_relu_3x3(new relu::Batch<>());
    const size_t conv2_relu_3x3_id = topology->add(conv2_relu_3x3);

    /* lrn(conv2/norm2): alpha=0.0001, beta=0.75, local_size=5 */
    SharedPtr<lrn::Batch<> > conv2_norm2(new lrn::Batch<>());
    conv2_norm2->parameter.kappa   = 1;
    conv2_norm2->parameter.nAdjust = 5;
    conv2_norm2->parameter.beta    = 0.75;
    conv2_norm2->parameter.alpha   = 0.0001 / conv2_norm2->parameter.nAdjust;
    const size_t conv2_norm2_id = topology->add(conv2_norm2);

    /* pooling(pool2/3x3_s2): 3x3 + 1x1s */
    SharedPtr<maximum_pooling2d::Batch<> > pool2_3x3_s2(new maximum_pooling2d::Batch<>(4));
    pool2_3x3_s2->parameter.kernelSizes = pooling2d::KernelSizes(3, 3);
    pool2_3x3_s2->parameter.strides     = pooling2d::Strides(2, 2);
    const size_t pool2_3x3_s2_id = topology->add(pool2_3x3_s2);

    /* inception module, convolution filters = (64, 96, 128, 16, 32, 32) */
    size_t inception1_start_id, inception1_end_id;
    training::TopologyPtr inception1 = configureInception(64, 96, 128, 16, 32, 32);
    inception1_end_id = topology->add(*inception1, inception1_start_id);

    /* inception module, convolution filters = (128, 128, 192, 32, 96, 64) */
    size_t inception2_start_id, inception2_end_id;
    training::TopologyPtr inception2 = configureInception(128, 128, 192, 32, 96, 64);
    inception2_end_id = topology->add(*inception2, inception2_start_id);

    /* pooling(pool3/3x3_s2): 3x3 + 2x2s */
    SharedPtr<maximum_pooling2d::Batch<> > pool3_3x3_s2(new maximum_pooling2d::Batch<>(4));
    pool3_3x3_s2->parameter.kernelSizes = pooling2d::KernelSizes(3, 3);
    pool3_3x3_s2->parameter.strides     = pooling2d::Strides(2, 2);
    const size_t pool3_3x3_s2_id = topology->add(pool3_3x3_s2);

    /* inception module, convolution filters = (192, 96, 208, 16, 48, 64) */
    size_t inception3_start_id, inception3_end_id;
    training::TopologyPtr inception3 = configureInception(192, 96, 208, 16, 48, 64);
    inception3_end_id = topology->add(*inception3, inception3_start_id);

    /* inception module, convolution filters = (160, 112, 224, 24, 64, 64) */
    size_t inception4_start_id, inception4_end_id;
    training::TopologyPtr inception4 = configureInception(160, 112, 224, 24, 64, 64, /* has loss branch */ true);
    inception4_end_id = topology->add(*inception4, inception4_start_id);

    /* loss branch 1 */
    size_t loss1_start_id, loss1_end_id;
    training::TopologyPtr loss1Branch = configureLossLayers();
    loss1_end_id = topology->add(*loss1Branch, loss1_start_id);

    /* inception module, convolution filters = (128, 128, 256, 24, 64, 64) */
    size_t inception5_start_id, inception5_end_id;
    training::TopologyPtr inception5 = configureInception(128, 128, 256, 24, 64, 64);
    inception5_end_id = topology->add(*inception5, inception5_start_id);

    /* inception module, convolution filters = (112, 144, 288, 32, 64, 64) */
    size_t inception6_start_id, inception6_end_id;
    training::TopologyPtr inception6 = configureInception(112, 144, 288, 32, 64, 64);
    inception6_end_id = topology->add(*inception6, inception6_start_id);

    /* inception module, convolution filters = (256, 160, 320, 32, 128, 128) */
    size_t inception7_start_id, inception7_end_id;
    training::TopologyPtr inception7 = configureInception(256, 160, 320, 32, 128, 128, /* has loss branch */ true);
    inception7_end_id = topology->add(*inception7, inception7_start_id);

    /* loss branch 2 */
    size_t loss2_start_id, loss2_end_id;
    training::TopologyPtr loss2Branch = configureLossLayers();
    loss2_end_id = topology->add(*loss2Branch, loss2_start_id);

    /* pooling(pool4_3x3_s2): 3x3 + 2x2s */
    SharedPtr<maximum_pooling2d::Batch<> > pool4_3x3_s2(new maximum_pooling2d::Batch<>(4));
    pool4_3x3_s2->parameter.kernelSizes = pooling2d::KernelSizes(3, 3);
    pool4_3x3_s2->parameter.strides     = pooling2d::Strides(2, 2);
    const size_t pool4_3x3_s2_id = topology->add(pool4_3x3_s2);

    /* inception module, convolution filters = (256, 160, 320, 32, 128, 128) */
    size_t inception8_start_id, inception8_end_id;
    training::TopologyPtr inception8 = configureInception(256, 160, 320, 32, 128, 128);
    inception8_end_id = topology->add(*inception8, inception8_start_id);

    /* inception module, convolution filters = (384, 192, 384, 48, 128, 128) */
    size_t inception9_start_id, inception9_end_id;
    training::TopologyPtr inception9 = configureInception(384, 192, 384, 48, 128, 128);
    inception9_end_id = topology->add(*inception9, inception9_start_id);

    /* pooling(pool5/7x7_s1): 7x7 + 1x1s */
    SharedPtr<average_pooling2d::Batch<> > pool5_7x7_s1(new average_pooling2d::Batch<>(4));
    pool5_7x7_s1->parameter.kernelSizes = pooling2d::KernelSizes(7, 7);
    pool5_7x7_s1->parameter.strides     = pooling2d::Strides(1, 1);
    const size_t pool5_7x7_s1_id = topology->add(pool5_7x7_s1);

    /* dropout(pool5/drop_7x7_s1): p = 0.4 */
    SharedPtr<dropout::Batch<> > pool5_drop_7x7_s1(new dropout::Batch<>());
    pool5_drop_7x7_s1->parameter.retainRatio = 0.4;
    const size_t pool5_drop_7x7_s1_id = topology->add(pool5_drop_7x7_s1);

    /* fullyconnected(loss3/classifier): n = 1000 */
    SharedPtr<fullyconnected::Batch<> > loss3_classifier(new fullyconnected::Batch<>(1000));
    loss3_classifier->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    loss3_classifier->parameter.biasesInitializer  = UniformInitializerPtr(new UniformInitializer(0.0, 0.0));
    const size_t loss3_classifier_id = topology->add(loss3_classifier);

    /* softmax + crossentropy loss (loss3/loss3) */
    SharedPtr<loss::softmax_cross::Batch<> > loss3_loss3(new loss::softmax_cross::Batch<>());
    const size_t loss3_end_id = topology->add(loss3_loss3);

    topology->get( conv1_7x7_s2_id          ).addNext( conv1_relu_7x7_id        );
    topology->get( conv1_relu_7x7_id        ).addNext( pool1_3x3_s2_id          );
    topology->get( pool1_3x3_s2_id          ).addNext( pool1_norm1_id           );
    topology->get( pool1_norm1_id           ).addNext( conv2_3x3_reduce_id      );
    topology->get( conv2_3x3_reduce_id      ).addNext( conv2_relu_3x3_reduce_id );
    topology->get( conv2_relu_3x3_reduce_id ).addNext( conv2_3x3_id             );
    topology->get( conv2_3x3_id             ).addNext( conv2_relu_3x3_id        );
    topology->get( conv2_relu_3x3_id        ).addNext( conv2_norm2_id           );
    topology->get( conv2_norm2_id           ).addNext( pool2_3x3_s2_id          );
    topology->get( pool2_3x3_s2_id          ).addNext( inception1_start_id      );
    topology->get( inception1_end_id        ).addNext( inception2_start_id      );
    topology->get( inception2_end_id        ).addNext( pool3_3x3_s2_id          );
    topology->get( pool3_3x3_s2_id          ).addNext( inception3_start_id      );
    topology->get( inception3_end_id        ).addNext( inception4_start_id      );
    topology->get( inception4_start_id      ).addNext( loss1_start_id           );
    topology->get( inception4_end_id        ).addNext( inception5_start_id      );
    topology->get( inception5_end_id        ).addNext( inception6_start_id      );
    topology->get( inception6_end_id        ).addNext( inception7_start_id      );
    topology->get( inception7_start_id      ).addNext( loss2_start_id           );
    topology->get( inception7_end_id        ).addNext( pool4_3x3_s2_id          );
    topology->get( pool4_3x3_s2_id          ).addNext( inception8_start_id      );
    topology->get( inception8_end_id        ).addNext( inception9_start_id      );
    topology->get( inception9_end_id        ).addNext( pool5_7x7_s1_id          );
    topology->get( pool5_7x7_s1_id          ).addNext( pool5_drop_7x7_s1_id     );
    topology->get( pool5_drop_7x7_s1_id     ).addNext( loss3_classifier_id      );
    topology->get( loss3_classifier_id      ).addNext( loss3_end_id             );

    return topology;
}

#endif
