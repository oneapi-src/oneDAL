/* file: lcn_parameter.cpp */
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
#include "neural_networks/layers/lcn/JLcnParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lcn_LcnParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lcn_LcnParameter_cInit
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new lcn::Parameter);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lcn_LcnParameter
 * Method:    cSetIndices
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lcn_LcnParameter_cSetIndices
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second)
{
    (((lcn::Parameter *)cParameter))->indices.dims[0] = first;
    (((lcn::Parameter *)cParameter))->indices.dims[1] = second;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lcn_LcnParameter
 * Method:    cGetKernelSize
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lcn_LcnParameter_cGetIndices
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(2);
    jlong tmp[2];
    tmp[0] = (jlong)((((lcn::Parameter *)cParameter))->indices.dims[0]);
    tmp[1] = (jlong)((((lcn::Parameter *)cParameter))->indices.dims[1]);
    env->SetLongArrayRegion(sizeArray, 0, 2, tmp);

    return sizeArray;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lcn_LcnParameter
 * Method:    cGetSumDimension
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lcn_LcnParameter_cGetSumDimension
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    NumericTablePtr *ntShPtr = new NumericTablePtr();
    *ntShPtr = (((lcn::Parameter *)cParameter))->sumDimension;
    return (jlong)ntShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lcn_LcnParameter
 * Method:    cSetSumDimension
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lcn_LcnParameter_cSetSumDimension
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong sumDimension)
{
    SerializationIfacePtr *ntShPtr = (SerializationIfacePtr *)sumDimension;
    (((lcn::Parameter *)cParameter))->sumDimension = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lcn_LcnParameter
 * Method:    cGetKernel
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lcn_LcnParameter_cGetKernel
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    TensorPtr *ntShPtr = new TensorPtr();
    *ntShPtr = (((lcn::Parameter *)cParameter))->kernel;
    return (jlong)ntShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lcn_LcnParameter
 * Method:    cSetKernel
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lcn_LcnParameter_cSetKernel
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong kernel)
{
    SerializationIfacePtr *ntShPtr = (SerializationIfacePtr *)kernel;
    (((lcn::Parameter *)cParameter))->kernel = staticPointerCast<Tensor, SerializationIface>(*ntShPtr);
}
