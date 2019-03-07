/* file: pooling3d_parameter.cpp */
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
#include "neural_networks/layers/pooling3d/JPooling3dParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling3d_Pooling3dParameter
 * Method:    cSetKernelSize
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling3d_Pooling3dParameter_cSetKernelSizes
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second, jlong third)
{
    (((pooling3d::Parameter *)cParameter))->kernelSizes.size[0] = first;
    (((pooling3d::Parameter *)cParameter))->kernelSizes.size[1] = second;
    (((pooling3d::Parameter *)cParameter))->kernelSizes.size[2] = third;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling3d_Pooling3dParameter
 * Method:    cSetStrides
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling3d_Pooling3dParameter_cSetStrides
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second, jlong third)
{
    (((pooling3d::Parameter *)cParameter))->strides.size[0] = first;
    (((pooling3d::Parameter *)cParameter))->strides.size[1] = second;
    (((pooling3d::Parameter *)cParameter))->strides.size[2] = third;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling3d_Pooling3dParameter
 * Method:    cSetSD
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling3d_Pooling3dParameter_cSetSD
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second, jlong third)
{
    (((pooling3d::Parameter *)cParameter))->indices.size[0] = first;
    (((pooling3d::Parameter *)cParameter))->indices.size[1] = second;
    (((pooling3d::Parameter *)cParameter))->indices.size[2] = third;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling3d_Pooling3dParameter
 * Method:    cSetPaddings
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling3d_Pooling3dParameter_cSetPaddings
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second, jlong third)
{
    (((pooling3d::Parameter *)cParameter))->paddings.size[0] = first;
    (((pooling3d::Parameter *)cParameter))->paddings.size[1] = second;
    (((pooling3d::Parameter *)cParameter))->paddings.size[2] = third;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling3d_Pooling3dParameter
 * Method:    cGetKernelSizes
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling3d_Pooling3dParameter_cGetKernelSizes
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(3);
    jlong tmp[3];
    tmp[0] = (jlong)((((pooling3d::Parameter *)cParameter))->kernelSizes.size[0]);
    tmp[1] = (jlong)((((pooling3d::Parameter *)cParameter))->kernelSizes.size[1]);
    tmp[2] = (jlong)((((pooling3d::Parameter *)cParameter))->kernelSizes.size[2]);
    env->SetLongArrayRegion(sizeArray, 0, 3, tmp);

    return sizeArray;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling3d_Pooling3dParameter
 * Method:    cGetStrides
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling3d_Pooling3dParameter_cGetStrides
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(3);
    jlong tmp[3];
    tmp[0] = (jlong)((((pooling3d::Parameter *)cParameter))->strides.size[0]);
    tmp[1] = (jlong)((((pooling3d::Parameter *)cParameter))->strides.size[1]);
    tmp[2] = (jlong)((((pooling3d::Parameter *)cParameter))->strides.size[2]);
    env->SetLongArrayRegion(sizeArray, 0, 3, tmp);

    return sizeArray;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling3d_Pooling3dParameter
 * Method:    cGetPaddings
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling3d_Pooling3dParameter_cGetPaddings
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(3);
    jlong tmp[3];
    tmp[0] = (jlong)((((pooling3d::Parameter *)cParameter))->paddings.size[0]);
    tmp[1] = (jlong)((((pooling3d::Parameter *)cParameter))->paddings.size[1]);
    tmp[2] = (jlong)((((pooling3d::Parameter *)cParameter))->paddings.size[2]);
    env->SetLongArrayRegion(sizeArray, 0, 3, tmp);

    return sizeArray;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling3d_Pooling3dParameter
 * Method:    cGetSD
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_pooling3d_Pooling3dParameter_cGetSD
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(3);
    jlong tmp[3];
    tmp[0] = (jlong)((((pooling3d::Parameter *)cParameter))->indices.size[0]);
    tmp[1] = (jlong)((((pooling3d::Parameter *)cParameter))->indices.size[1]);
    tmp[2] = (jlong)((((pooling3d::Parameter *)cParameter))->indices.size[2]);
    env->SetLongArrayRegion(sizeArray, 0, 3, tmp);

    return sizeArray;
}
