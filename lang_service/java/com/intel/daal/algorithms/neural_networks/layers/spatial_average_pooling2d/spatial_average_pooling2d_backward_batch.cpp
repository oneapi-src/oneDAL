/* file: spatial_average_pooling2d_backward_batch.cpp */
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
#include "neural_networks/layers/spatial_average_pooling2d/JSpatialAveragePooling2dBackwardBatch.h"

#include "daal.h"

#include "common_helpers.h"

using namespace daal;
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_average_1pooling2d_SpatialAveragePooling2dBackwardBatch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1average_1pooling2d_SpatialAveragePooling2dBackwardBatch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method, jlong pyramidHeigh, jlong nDim)
{
    return jniBatch<spatial_average_pooling2d::Method, spatial_average_pooling2d::backward::Batch, spatial_average_pooling2d::defaultDense>::
           newObj(prec, method, pyramidHeigh, nDim);
}


/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_average_1pooling2d_SpatialAveragePooling2dBackwardBatch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1average_1pooling2d_SpatialAveragePooling2dBackwardBatch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<spatial_average_pooling2d::Method, spatial_average_pooling2d::backward::Batch, spatial_average_pooling2d::defaultDense>::
           getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_average_1pooling2d_SpatialAveragePooling2dBackwardBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1average_1pooling2d_SpatialAveragePooling2dBackwardBatch_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<spatial_average_pooling2d::Method, spatial_average_pooling2d::backward::Batch, spatial_average_pooling2d::defaultDense>::
           getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_average_1pooling2d_SpatialAveragePooling2dBackwardBatch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1average_1pooling2d_SpatialAveragePooling2dBackwardBatch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<spatial_average_pooling2d::Method, spatial_average_pooling2d::backward::Batch, spatial_average_pooling2d::defaultDense>::
           getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_average_1pooling2d_SpatialAveragePooling2dBackwardBatch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1average_1pooling2d_SpatialAveragePooling2dBackwardBatch_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resAddr)
{
    jniBatch<spatial_average_pooling2d::Method, spatial_average_pooling2d::backward::Batch, spatial_average_pooling2d::defaultDense>::
    setResult<spatial_average_pooling2d::backward::Result>(prec, method, algAddr, resAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_average_1pooling2d_SpatialAveragePooling2dBackwardBatch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1average_1pooling2d_SpatialAveragePooling2dBackwardBatch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<spatial_average_pooling2d::Method, spatial_average_pooling2d::backward::Batch, spatial_average_pooling2d::defaultDense>::
           getClone(prec, method, algAddr);
}
