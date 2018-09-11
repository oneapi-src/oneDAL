/* file: softmax_cross_parameter.cpp */
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
#include "neural_networks/layers/softmax_cross/JSoftmaxCrossParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers::loss;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_softmax_cross_SoftmaxCrossParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_softmax_1cross_SoftmaxCrossParameter_cInit
  (JNIEnv *env, jobject thisObj)
{
    return (jlong)(new softmax_cross::Parameter);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_softmax_cross_SoftmaxCrossParameter
 * Method:    cGetAccuracyThreshold
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_softmax_1cross_SoftmaxCrossParameter_cGetAccuracyThreshold
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (((softmax_cross::Parameter *)cParameter))->accuracyThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_softmax_cross_SoftmaxCrossParameter
 * Method:    cSetAccuracyThreshold
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_softmax_1cross_SoftmaxCrossParameter_cSetAccuracyThreshold
  (JNIEnv *env, jobject thisObj, jlong cParameter, jdouble accuracyThreshold)
{
    (((softmax_cross::Parameter *)cParameter))->accuracyThreshold = accuracyThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_softmax_cross_SoftmaxCrossParameter
 * Method:    cGetDimension
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_softmax_1cross_SoftmaxCrossParameter_cGetDimension
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (((softmax_cross::Parameter *)cParameter))->dimension;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_softmax_cross_SoftmaxCrossParameter
 * Method:    cSetDimension
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_softmax_1cross_SoftmaxCrossParameter_cSetDimension
  (JNIEnv *env, jobject thisObj, jlong cParameter, jdouble dimension)
{
    (((softmax_cross::Parameter *)cParameter))->dimension = dimension;
}
