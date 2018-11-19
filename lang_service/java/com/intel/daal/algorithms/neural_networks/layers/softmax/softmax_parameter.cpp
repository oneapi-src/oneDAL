/* file: softmax_parameter.cpp */
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
#include "neural_networks/layers/softmax/JSoftmaxParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_softmax_SoftmaxParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_softmax_SoftmaxParameter_cInit
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new softmax::Parameter);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_softmax_SoftmaxParameter
 * Method:    cGetDimension
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_softmax_SoftmaxParameter_cGetDimension
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((softmax::Parameter *)cParameter))->dimension);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_softmax_SoftmaxParameter
 * Method:    cSetDimension
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_softmax_SoftmaxParameter_cSetDimension
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong dimension)
{
    (((softmax::Parameter *)cParameter))->dimension = (size_t)dimension;
}
