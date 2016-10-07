/* file: spatial_pooling2d_parameter.cpp */
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
#include "neural_networks/layers/spatial_pooling2d/JSpatialPooling2dParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling2d_SpatialPooling2dParameter
 * Method:    cSetIndices
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1pooling2d_SpatialPooling2dParameter_cSetIndices
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second)
{
    (((spatial_pooling2d::Parameter *)cParameter))->indices.size[0] = first;
    (((spatial_pooling2d::Parameter *)cParameter))->indices.size[1] = second;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling2d_SpatialPooling2dParameter
 * Method:    cSetPyramidHeight
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1pooling2d_SpatialPooling2dParameter_cSetPyramidHeight
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong pyramidHeight)
{
    (((spatial_pooling2d::Parameter *)cParameter))->pyramidHeight = pyramidHeight;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling2d_SpatialPooling2dParameter
 * Method:    cGetPyramidHeight
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1pooling2d_SpatialPooling2dParameter_cGetPyramidHeight
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((spatial_pooling2d::Parameter *)cParameter))->pyramidHeight);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling2d_SpatialPooling2dParameter
 * Method:    cGetIndices
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1pooling2d_SpatialPooling2dParameter_cGetIndices
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(2);
    jlong tmp[2];
    tmp[0] = (jlong)((((spatial_pooling2d::Parameter *)cParameter))->indices.size[0]);
    tmp[1] = (jlong)((((spatial_pooling2d::Parameter *)cParameter))->indices.size[1]);
    env->SetLongArrayRegion(sizeArray, 0, 2, tmp);

    return sizeArray;
}
