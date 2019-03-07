/* file: stochastic_pooling2d_parameter.cpp */
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
#include "neural_networks/layers/stochastic_pooling2d/JStochasticPooling2dParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_stochastic_1pooling2d_StochasticPooling2dParameter
 * Method:    cGetPredictionStage
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_stochastic_1pooling2d_StochasticPooling2dParameter_cGetPredictionStage
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (((stochastic_pooling2d::Parameter *)cParameter))->predictionStage;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_stochastic_1pooling2d_StochasticPooling2dParameter
 * Method:    cSetPredictionStage
 * Signature: (JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_stochastic_1pooling2d_StochasticPooling2dParameter_cSetPredictionStage
  (JNIEnv *env, jobject thisObj, jlong cParameter, jboolean predictionStage)
{
    (((stochastic_pooling2d::Parameter *)cParameter))->predictionStage = predictionStage;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_stochastic_1pooling2d_StochasticPooling2dParameter
 * Method:    cGetSeed
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_stochastic_1pooling2d_StochasticPooling2dParameter_cGetSeed
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((stochastic_pooling2d::Parameter *)cParameter))->seed);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_stochastic_1pooling2d_StochasticPooling2dParameter
 * Method:    cSetSeed
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_stochastic_1pooling2d_StochasticPooling2dParameter_cSetSeed
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong seed)
{
    (((stochastic_pooling2d::Parameter *)cParameter))->seed = (size_t)seed;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_stochastic_1pooling2d_StochasticPooling2dParameter
 * Method:    cSetEngine
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_stochastic_1pooling2d_StochasticPooling2dParameter_cSetEngine
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong engineAddr)
{
    (((stochastic_pooling2d::Parameter *)cParameter))->engine = staticPointerCast<engines::BatchBase, AlgorithmIface> (*(SharedPtr<AlgorithmIface> *)engineAddr);
}
