/* file: lcn_parameter.cpp */
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
    SharedPtr<NumericTable> *ntShPtr = new SharedPtr<NumericTable>();
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
    SharedPtr<SerializationIface> *ntShPtr = (SharedPtr<SerializationIface> *)sumDimension;
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
    SharedPtr<Tensor> *ntShPtr = new SharedPtr<Tensor>();
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
    SharedPtr<SerializationIface> *ntShPtr = (SharedPtr<SerializationIface> *)kernel;
    (((lcn::Parameter *)cParameter))->kernel = staticPointerCast<Tensor, SerializationIface>(*ntShPtr);
}
