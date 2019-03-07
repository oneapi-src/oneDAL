/* file: average_pooling2d_batch.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
*******************************************************************************/

#include <jni.h>
#include "neural_networks/layers/average_pooling2d/JAveragePooling2dBatch.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_average_1pooling2d_AveragePooling2dBatch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_average_1pooling2d_AveragePooling2dBatch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method, jlong nDim)
{
    return jniBatchLayer<average_pooling2d::Method, average_pooling2d::Batch, average_pooling2d::defaultDense>::
           newObj(prec, method, nDim);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_average_1pooling2d_AveragePooling2dBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_average_1pooling2d_AveragePooling2dBatch_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatchLayer<average_pooling2d::Method, average_pooling2d::Batch, average_pooling2d::defaultDense>::
           getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_average_1pooling2d_AveragePooling2dBatch
 * Method:    cGetForwardLayer
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_average_1pooling2d_AveragePooling2dBatch_cGetForwardLayer
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatchLayer<average_pooling2d::Method, average_pooling2d::Batch, average_pooling2d::defaultDense>::
           getForwardLayer(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_average_1pooling2d_AveragePooling2dBatch
 * Method:    cGetBackwardLayer
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_average_1pooling2d_AveragePooling2dBatch_cGetBackwardLayer
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatchLayer<average_pooling2d::Method, average_pooling2d::Batch, average_pooling2d::defaultDense>::
           getBackwardLayer(prec, method, algAddr);
}
