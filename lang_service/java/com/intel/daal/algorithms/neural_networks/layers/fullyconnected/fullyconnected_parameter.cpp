/* file: fullyconnected_parameter.cpp */
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
#include "neural_networks/layers/fullyconnected/JFullyConnectedParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_fullyconnected_FullyConnectedParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_fullyconnected_FullyConnectedParameter_cInit
  (JNIEnv *env, jobject thisObj, jlong nOutputs)
{
    return (jlong)(new fullyconnected::Parameter((size_t)nOutputs));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_fullyconnected_FullyConnectedParameter
 * Method:    cGetNOutputs
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_fullyconnected_FullyConnectedParameter_cGetNOutputs
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((fullyconnected::Parameter *)cParameter))->nOutputs);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_fullyconnected_FullyConnectedParameter
 * Method:    cSetNOutputs
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_fullyconnected_FullyConnectedParameter_cSetNOutputs
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong nOutputs)
{
    (((fullyconnected::Parameter *)cParameter))->nOutputs = (size_t)nOutputs;
}
