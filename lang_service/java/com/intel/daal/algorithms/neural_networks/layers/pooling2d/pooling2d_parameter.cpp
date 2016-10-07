/* file: pooling2d_parameter.cpp */
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
#include "neural_networks/layers/pooling2d/JPooling2dParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling2d_Pooling2dParameter
 * Method:    cSetKernelSizes
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling2d_Pooling2dParameter_cSetKernelSizes
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second)
{
    (((pooling2d::Parameter *)cParameter))->kernelSizes.size[0] = first;
    (((pooling2d::Parameter *)cParameter))->kernelSizes.size[1] = second;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling2d_Pooling2dParameter
 * Method:    cSetStrides
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling2d_Pooling2dParameter_cSetStrides
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second)
{
    (((pooling2d::Parameter *)cParameter))->strides.size[0] = first;
    (((pooling2d::Parameter *)cParameter))->strides.size[1] = second;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling2d_Pooling2dParameter
 * Method:    cSetSD
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling2d_Pooling2dParameter_cSetSD
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second)
{
    (((pooling2d::Parameter *)cParameter))->indices.size[0] = first;
    (((pooling2d::Parameter *)cParameter))->indices.size[1] = second;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling2d_Pooling2dParameter
 * Method:    cSetPaddings
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling2d_Pooling2dParameter_cSetPaddings
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second)
{
    (((pooling2d::Parameter *)cParameter))->paddings.size[0] = first;
    (((pooling2d::Parameter *)cParameter))->paddings.size[1] = second;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling2d_Pooling2dParameter
 * Method:    cGetKernelSizes
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling2d_Pooling2dParameter_cGetKernelSizes
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(2);
    jlong tmp[2];
    tmp[0] = (jlong)((((pooling2d::Parameter *)cParameter))->kernelSizes.size[0]);
    tmp[1] = (jlong)((((pooling2d::Parameter *)cParameter))->kernelSizes.size[1]);
    env->SetLongArrayRegion(sizeArray, 0, 2, tmp);

    return sizeArray;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling2d_Pooling2dParameter
 * Method:    cGetStrides
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling2d_Pooling2dParameter_cGetStrides
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(2);
    jlong tmp[2];
    tmp[0] = (jlong)((((pooling2d::Parameter *)cParameter))->strides.size[0]);
    tmp[1] = (jlong)((((pooling2d::Parameter *)cParameter))->strides.size[1]);
    env->SetLongArrayRegion(sizeArray, 0, 2, tmp);

    return sizeArray;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling2d_Pooling2dParameter
 * Method:    cGetPaddings
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling2d_Pooling2dParameter_cGetPaddings
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(2);
    jlong tmp[2];
    tmp[0] = (jlong)((((pooling2d::Parameter *)cParameter))->paddings.size[0]);
    tmp[1] = (jlong)((((pooling2d::Parameter *)cParameter))->paddings.size[1]);
    env->SetLongArrayRegion(sizeArray, 0, 2, tmp);

    return sizeArray;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling2d_Pooling2dParameter
 * Method:    cGetSD
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling2d_Pooling2dParameter_cGetSD
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(2);
    jlong tmp[2];
    tmp[0] = (jlong)((((pooling2d::Parameter *)cParameter))->indices.size[0]);
    tmp[1] = (jlong)((((pooling2d::Parameter *)cParameter))->indices.size[1]);
    env->SetLongArrayRegion(sizeArray, 0, 2, tmp);

    return sizeArray;
}
