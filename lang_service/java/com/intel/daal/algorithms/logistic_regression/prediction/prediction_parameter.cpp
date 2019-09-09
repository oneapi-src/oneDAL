/* file: prediction_parameter.cpp */
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

#include "daal.h"
#include "com_intel_daal_algorithms_logistic_regression_prediction_PredictionParameter.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_prediction_PredictionParameter
 * Method:    cParInit
 * Signature:()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionParameter_cParInit
(JNIEnv *env, jobject thisObj)
{
    return(jlong)new logistic_regression::prediction::Parameter();
}

/*
 * Class:     com_intel_daal_algorithms_logistic_1regression_PredictionParameter
 * Method:    cSetSeed
 * Signature:(JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionParameter_cSetResultsToCompute
(JNIEnv *env, jobject thisObj, jlong parAddr, jlong resToCompute)
{
    (*(logistic_regression::prediction::Parameter *)parAddr).resultsToCompute = resToCompute;
}

/*
 * Class:     com_intel_daal_algorithms_logistic_1regression_PredictionParameter
 * Method:    cGetSeed
 * Signature:(J)I
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionParameter_cGetResultsToCompute
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return(jlong)(*(logistic_regression::prediction::Parameter *)parAddr).resultsToCompute;
}
