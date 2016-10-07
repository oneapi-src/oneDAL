/* file: spatial_average_pooling2d_backward_batch.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
