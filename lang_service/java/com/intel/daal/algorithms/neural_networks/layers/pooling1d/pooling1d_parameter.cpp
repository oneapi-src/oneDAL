/* file: pooling1d_parameter.cpp */
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
#include "neural_networks/layers/pooling1d/JPooling1dParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling1d_Pooling1dParameter
 * Method:    cSetKernelSize
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling1d_Pooling1dParameter_cSetKernelSize
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first)
{
    (((pooling1d::Parameter *)cParameter))->kernelSize.size[0] = first;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling1d_Pooling1dParameter
 * Method:    cSetStride
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling1d_Pooling1dParameter_cSetStride
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first)
{
    (((pooling1d::Parameter *)cParameter))->stride.size[0] = first;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling1d_Pooling1dParameter
 * Method:    cSetSD
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling1d_Pooling1dParameter_cSetSD
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first)
{
    (((pooling1d::Parameter *)cParameter))->index.size[0] = first;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling1d_Pooling1dParameter
 * Method:    cSetPadding
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling1d_Pooling1dParameter_cSetPadding
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first)
{
    (((pooling1d::Parameter *)cParameter))->padding.size[0] = first;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling1d_Pooling1dParameter
 * Method:    cGetKernelSize
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling1d_Pooling1dParameter_cGetKernelSize
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(1);
    jlong tmp[1];
    tmp[0] = (jlong)((((pooling1d::Parameter *)cParameter))->kernelSize.size[0]);
    env->SetLongArrayRegion(sizeArray, 0, 1, tmp);

    return sizeArray;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling1d_Pooling1dParameter
 * Method:    cGetKernelSize
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling1d_Pooling1dParameter_cGetStride
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(1);
    jlong tmp[1];
    tmp[0] = (jlong)((((pooling1d::Parameter *)cParameter))->stride.size[0]);
    env->SetLongArrayRegion(sizeArray, 0, 1, tmp);

    return sizeArray;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling1d_Pooling1dParameter
 * Method:    cGetKernelSize
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling1d_Pooling1dParameter_cGetPadding
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(1);
    jlong tmp[1];
    tmp[0] = (jlong)((((pooling1d::Parameter *)cParameter))->padding.size[0]);
    env->SetLongArrayRegion(sizeArray, 0, 1, tmp);

    return sizeArray;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling1d_Pooling1dParameter
 * Method:    cGetKernelSize
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling1d_Pooling1dParameter_cGetSD
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(1);
    jlong tmp[1];
    tmp[0] = (jlong)((((pooling1d::Parameter *)cParameter))->index.size[0]);
    env->SetLongArrayRegion(sizeArray, 0, 1, tmp);

    return sizeArray;
}
