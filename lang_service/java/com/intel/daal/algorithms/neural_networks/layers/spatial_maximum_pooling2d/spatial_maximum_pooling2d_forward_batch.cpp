/* file: spatial_maximum_pooling2d_forward_batch.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
#include "neural_networks/layers/spatial_maximum_pooling2d/JSpatialMaximumPooling2dForwardBatch.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_maximum_1pooling2d_SpatialMaximumPooling2dForwardBatch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1maximum_1pooling2d_SpatialMaximumPooling2dForwardBatch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method, jlong pyramidHeigh, jlong nDim)
{
    return jniBatch<spatial_maximum_pooling2d::Method, spatial_maximum_pooling2d::forward::Batch, spatial_maximum_pooling2d::defaultDense>::
           newObj(prec, method, pyramidHeigh, nDim);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_maximum_1pooling2d_SpatialMaximumPooling2dForwardBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1maximum_1pooling2d_SpatialMaximumPooling2dForwardBatch_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<spatial_maximum_pooling2d::Method, spatial_maximum_pooling2d::forward::Batch, spatial_maximum_pooling2d::defaultDense>::
           getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_maximum_1pooling2d_SpatialMaximumPooling2dForwardBatch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1maximum_1pooling2d_SpatialMaximumPooling2dForwardBatch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<spatial_maximum_pooling2d::Method, spatial_maximum_pooling2d::forward::Batch, spatial_maximum_pooling2d::defaultDense>::
           getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_maximum_1pooling2d_SpatialMaximumPooling2dForwardBatch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1maximum_1pooling2d_SpatialMaximumPooling2dForwardBatch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<spatial_maximum_pooling2d::Method, spatial_maximum_pooling2d::forward::Batch, spatial_maximum_pooling2d::defaultDense>::
           getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_maximum_1pooling2d_SpatialMaximumPooling2dForwardBatch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1maximum_1pooling2d_SpatialMaximumPooling2dForwardBatch_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resAddr)
{
    jniBatch<spatial_maximum_pooling2d::Method, spatial_maximum_pooling2d::forward::Batch, spatial_maximum_pooling2d::defaultDense>::
    setResult<spatial_maximum_pooling2d::forward::Result>(prec, method, algAddr, resAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_maximum_1pooling2d_SpatialMaximumPooling2dForwardBatch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1maximum_1pooling2d_SpatialMaximumPooling2dForwardBatch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<spatial_maximum_pooling2d::Method, spatial_maximum_pooling2d::forward::Batch, spatial_maximum_pooling2d::defaultDense>::
           getClone(prec, method, algAddr);
}
