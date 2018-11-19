/* file: parameter.cpp */
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
#include "neural_networks/layers/JParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_Parameter
 * Method:    cSetWeightsInitializer
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_Parameter_cSetWeightsInitializer
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong cInitializer)
{
    (((layers::Parameter *)cParameter))->weightsInitializer =
        staticPointerCast<initializers::InitializerIface, AlgorithmIface>(*((SharedPtr<AlgorithmIface> *)cInitializer));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_Parameter
 * Method:    cSetBiasesInitializer
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_Parameter_cSetBiasesInitializer
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong cInitializer)
{
    (((layers::Parameter *)cParameter))->biasesInitializer =
        staticPointerCast<initializers::InitializerIface, AlgorithmIface>(*((SharedPtr<AlgorithmIface> *)cInitializer));
}
/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_Parameter
 * Method:    cGetWeightsAndBiasesInitializationFlag
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_Parameter_cGetWeightsAndBiasesInitializationFlag
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (((layers::Parameter *)cParameter))->weightsAndBiasesInitialized;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_Parameter
 * Method:    cSetWeightsAndBiasesInitializationFlag
 * Signature: (JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_Parameter_cSetWeightsAndBiasesInitializationFlag
  (JNIEnv *env, jobject thisObj, jlong cParameter, jboolean weightsAndBiasesInitialized)
{
    (((layers::Parameter *)cParameter))->weightsAndBiasesInitialized = weightsAndBiasesInitialized;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_Parameter
 * Method:    cGetPredictionStage
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_Parameter_cGetPredictionStage
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (((layers::Parameter *)cParameter))->predictionStage;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_Parameter
 * Method:    cSetPredictionStage
 * Signature: (JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_Parameter_cSetPredictionStage
  (JNIEnv *env, jobject thisObj, jlong cParameter, jboolean predictionStage)
{
    (((layers::Parameter *)cParameter))->predictionStage = predictionStage;
}
