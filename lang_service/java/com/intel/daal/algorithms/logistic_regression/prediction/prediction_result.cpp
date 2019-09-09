/* file: prediction_result.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#include "com_intel_daal_algorithms_logistic_regression_prediction_PredictionResult.h"

#include "daal.h"

#include "common_helpers.h"

#include "com_intel_daal_algorithms_classifier_prediction_PredictionResultId.h"
#define predictionResultId com_intel_daal_algorithms_classifier_prediction_PredictionResultId_Prediction
#include "com_intel_daal_algorithms_logistic_regression_prediction_PredictionResultNumericTableId.h"
#define probabilitiesId    com_intel_daal_algorithms_logistic_regression_prediction_PredictionResultNumericTableId_probabilitiesValue
#define logProbabilitiesId com_intel_daal_algorithms_logistic_regression_prediction_PredictionResultNumericTableId_logProbabilitiesValue

USING_COMMON_NAMESPACES();

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_prediction_PredictionResult
 * Method:    cGetValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionResult_cGetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if (id == predictionResultId)
    {
        return jniArgument<logistic_regression::prediction::Result>::get<classifier::prediction::ResultId, NumericTable>(resAddr, id);
    }
    else if (id == probabilitiesId || id == logProbabilitiesId)
    {
        return jniArgument<logistic_regression::prediction::Result>::get<logistic_regression::prediction::ResultNumericTableId, NumericTable>(resAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_prediction_PredictionResult
 * Method:    cSetValue
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionResult_cSetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    if (id == predictionResultId)
    {
        jniArgument<logistic_regression::prediction::Result>::set<classifier::prediction::ResultId, NumericTable>(resAddr, id, ntAddr);
    }
    else if (id == probabilitiesId || id == logProbabilitiesId)
    {
        jniArgument<logistic_regression::prediction::Result>::set<logistic_regression::prediction::ResultNumericTableId, NumericTable>(resAddr, id, ntAddr);
    }
}
