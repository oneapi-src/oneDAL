/* file: batch_normalization_parameter.cpp */
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
#include "neural_networks/layers/batch_normalization/JBatchNormalizationParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_batch_1normalization_BatchNormalizationParameter_cInit
  (JNIEnv *env, jobject thisObj)
{
    return (jlong)(new batch_normalization::Parameter);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationParameter
 * Method:    cGetAlpha
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_batch_1normalization_BatchNormalizationParameter_cGetAlpha
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (((batch_normalization::Parameter *)cParameter))->alpha;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationParameter
 * Method:    cSetAlpha
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_batch_1normalization_BatchNormalizationParameter_cSetAlpha
  (JNIEnv *env, jobject thisObj, jlong cParameter, jdouble alpha)
{
    (((batch_normalization::Parameter *)cParameter))->alpha = alpha;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationParameter
 * Method:    cGetEpsilon
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_batch_1normalization_BatchNormalizationParameter_cGetEpsilon
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (((batch_normalization::Parameter *)cParameter))->epsilon;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationParameter
 * Method:    cSetEpsilon
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_batch_1normalization_BatchNormalizationParameter_cSetEpsilon
  (JNIEnv *env, jobject thisObj, jlong cParameter, jdouble epsilon)
{
    (((batch_normalization::Parameter *)cParameter))->epsilon = epsilon;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationParameter
 * Method:    cGetDimension
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_batch_1normalization_BatchNormalizationParameter_cGetDimension
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((batch_normalization::Parameter *)cParameter))->dimension);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationParameter
 * Method:    cSetDimension
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_batch_1normalization_BatchNormalizationParameter_cSetDimension
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong dimension)
{
    (((batch_normalization::Parameter *)cParameter))->dimension = (size_t)dimension;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationParameter
 * Method:    cGetPredictionStage
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_batch_1normalization_BatchNormalizationParameter_cGetPredictionStage
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (((batch_normalization::Parameter *)cParameter))->predictionStage;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationParameter
 * Method:    cSetPredictionStage
 * Signature: (JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_batch_1normalization_BatchNormalizationParameter_cSetPredictionStage
  (JNIEnv *env, jobject thisObj, jlong cParameter, jboolean predictionStage)
{
    (((batch_normalization::Parameter *)cParameter))->predictionStage = predictionStage;
}
