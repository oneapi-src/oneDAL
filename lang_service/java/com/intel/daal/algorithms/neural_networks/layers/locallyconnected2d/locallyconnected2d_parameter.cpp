/* file: locallyconnected2d_parameter.cpp */
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
#include "neural_networks/layers/locallyconnected2d/JLocallyConnected2dParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_locallyconnected2d_LocallyConnected2dParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_locallyconnected2d_LocallyConnected2dParameter_cInit
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new locallyconnected2d::Parameter);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_locallyconnected2d_LocallyConnected2dParameter
 * Method:    cGetGroupDimension
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_locallyconnected2d_LocallyConnected2dParameter_cGetGroupDimension
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((locallyconnected2d::Parameter *)cParameter))->groupDimension);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_locallyconnected2d_LocallyConnected2dParameter
 * Method:    cSetGroupDimension
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_locallyconnected2d_LocallyConnected2dParameter_cSetGroupDimension
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong groupDimension)
{
    (((locallyconnected2d::Parameter *)cParameter))->groupDimension = (size_t)groupDimension;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_locallyconnected2d_LocallyConnected2dParameter
 * Method:    cgetNKernels
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_locallyconnected2d_LocallyConnected2dParameter_cgetNKernels
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((locallyconnected2d::Parameter *)cParameter))->nKernels);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_locallyconnected2d_LocallyConnected2dParameter
 * Method:    csetNKernels
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_locallyconnected2d_LocallyConnected2dParameter_csetNKernels
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong nKernels)
{
    (((locallyconnected2d::Parameter *)cParameter))->nKernels = (size_t)nKernels;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_locallyconnected2d_LocallyConnected2dParameter
 * Method:    cgetNGroups
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_locallyconnected2d_LocallyConnected2dParameter_cgetNGroups
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((locallyconnected2d::Parameter *)cParameter))->nGroups);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_locallyconnected2d_LocallyConnected2dParameter
 * Method:    csetNGroups
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_locallyconnected2d_LocallyConnected2dParameter_csetNGroups
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong nGroups)
{
    (((locallyconnected2d::Parameter *)cParameter))->nGroups = (size_t)nGroups;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_locallyconnected2d_LocallyConnected2dParameter
 * Method:    cSetKernelSizes
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_locallyconnected2d_LocallyConnected2dParameter_cSetKernelSizes
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second)
{
    (((locallyconnected2d::Parameter *)cParameter))->kernelSizes.size[0] = first;
    (((locallyconnected2d::Parameter *)cParameter))->kernelSizes.size[1] = second;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_locallyconnected2d_LocallyConnected2dParameter
 * Method:    cSetStrides
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_locallyconnected2d_LocallyConnected2dParameter_cSetStrides
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second)
{
    (((locallyconnected2d::Parameter *)cParameter))->strides.size[0] = first;
    (((locallyconnected2d::Parameter *)cParameter))->strides.size[1] = second;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_locallyconnected2d_LocallyConnected2dParameter
 * Method:    cSetIndices
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_locallyconnected2d_LocallyConnected2dParameter_cSetIndices
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second)
{
    (((locallyconnected2d::Parameter *)cParameter))->indices.dims[0] = first;
    (((locallyconnected2d::Parameter *)cParameter))->indices.dims[1] = second;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_locallyconnected2d_LocallyConnected2dParameter
 * Method:    cSetPaddings
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_locallyconnected2d_LocallyConnected2dParameter_cSetPaddings
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second)
{
    (((locallyconnected2d::Parameter *)cParameter))->paddings.size[0] = first;
    (((locallyconnected2d::Parameter *)cParameter))->paddings.size[1] = second;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_locallyconnected2d_LocallyConnected2dParameter
 * Method:    cGetKernelSizes
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_locallyconnected2d_LocallyConnected2dParameter_cGetKernelSizes
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(2);
    jlong tmp[2];
    tmp[0] = (jlong)((((locallyconnected2d::Parameter *)cParameter))->kernelSizes.size[0]);
    tmp[1] = (jlong)((((locallyconnected2d::Parameter *)cParameter))->kernelSizes.size[1]);
    env->SetLongArrayRegion(sizeArray, 0, 2, tmp);

    return sizeArray;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_locallyconnected2d_LocallyConnected2dParameter
 * Method:    cGetStrides
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_locallyconnected2d_LocallyConnected2dParameter_cGetStrides
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(2);
    jlong tmp[2];
    tmp[0] = (jlong)((((locallyconnected2d::Parameter *)cParameter))->strides.size[0]);
    tmp[1] = (jlong)((((locallyconnected2d::Parameter *)cParameter))->strides.size[1]);
    env->SetLongArrayRegion(sizeArray, 0, 2, tmp);

    return sizeArray;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_locallyconnected2d_LocallyConnected2dParameter
 * Method:    cGetPaddings
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_locallyconnected2d_LocallyConnected2dParameter_cGetPaddings
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(2);
    jlong tmp[2];
    tmp[0] = (jlong)((((locallyconnected2d::Parameter *)cParameter))->paddings.size[0]);
    tmp[1] = (jlong)((((locallyconnected2d::Parameter *)cParameter))->paddings.size[1]);
    env->SetLongArrayRegion(sizeArray, 0, 2, tmp);

    return sizeArray;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_locallyconnected2d_LocallyConnected2dParameter
 * Method:    cGetIndices
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_locallyconnected2d_LocallyConnected2dParameter_cGetIndices
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(2);
    jlong tmp[2];
    tmp[0] = (jlong)((((locallyconnected2d::Parameter *)cParameter))->indices.dims[0]);
    tmp[1] = (jlong)((((locallyconnected2d::Parameter *)cParameter))->indices.dims[1]);
    env->SetLongArrayRegion(sizeArray, 0, 2, tmp);

    return sizeArray;
}
