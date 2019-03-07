/* file: prelu_parameter.cpp */
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
#include "neural_networks/layers/prelu/JPreluParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_prelu_PreluParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_prelu_PreluParameter_cInit
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new prelu::Parameter);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_prelu_PreluParameter
 * Method:    cGetDimension
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_prelu_PreluParameter_cGetDataDimension
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((prelu::Parameter *)cParameter))->dataDimension);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_prelu_PreluParameter
 * Method:    cSetDimension
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_prelu_PreluParameter_cSetDataDimension
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong dataDimension)
{
    (((prelu::Parameter *)cParameter))->dataDimension = (size_t)dataDimension;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_prelu_PreluParameter
 * Method:    cGetDimension
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_prelu_PreluParameter_cgetWeightsDimension
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((prelu::Parameter *)cParameter))->weightsDimension);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_prelu_PreluParameter
 * Method:    cSetDimension
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_prelu_PreluParameter_csetWeightsDimension
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong weightsDimension)
{
    (((prelu::Parameter *)cParameter))->weightsDimension = (size_t)weightsDimension;
}
