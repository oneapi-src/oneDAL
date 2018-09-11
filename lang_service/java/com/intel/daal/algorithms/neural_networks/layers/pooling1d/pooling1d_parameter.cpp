/* file: pooling1d_parameter.cpp */
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
