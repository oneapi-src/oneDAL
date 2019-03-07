/* file: split_parameter.cpp */
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
#include "neural_networks/layers/split/JSplitParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitParameter_cInit
  (JNIEnv *env, jobject thisObj)
{
    return (jlong)(new split::Parameter);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitParameter
 * Method:    cGetNOutputs
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitParameter_cGetNOutputs
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((split::Parameter *)cParameter))->nOutputs);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitParameter
 * Method:    cSetNOutputs
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitParameter_cSetNOutputs
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong nOutputs)
{
    (((split::Parameter *)cParameter))->nOutputs = (size_t)nOutputs;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitParameter
 * Method:    cGetNInputs
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitParameter_cGetNInputs
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((split::Parameter *)cParameter))->nInputs);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitParameter
 * Method:    cGetNInputs
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitParameter_cSetNInputs
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong nInputs)
{
    (((split::Parameter *)cParameter))->nInputs = (size_t)nInputs;
}
