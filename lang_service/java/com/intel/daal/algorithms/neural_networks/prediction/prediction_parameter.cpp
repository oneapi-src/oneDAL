/* file: prediction_parameter.cpp */
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
#include "neural_networks/prediction/JPredictionParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionParameter_cInit
  (JNIEnv *env, jobject thisObj)
{
    return (jlong)(new prediction::Parameter());
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionParameter
 * Method:    cGetBatchSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionParameter_cGetBatchSize
  (JNIEnv *env, jobject thisObj, jlong addr)
{
    return (jlong)(((prediction::Parameter *)addr)->batchSize);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionParameter
 * Method:    cSetBatchSize
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionParameter_cSetBatchSize
  (JNIEnv *env, jobject thisObj, jlong addr, jlong batchSize)
{
    ((prediction::Parameter *)addr)->batchSize = batchSize;
}
