/* file: prediction_input.cpp */
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

#include "com_intel_daal_algorithms_logistic_regression_prediction_PredictionInput.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::logistic_regression::prediction;

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_prediction_PredictionInput
 * Method:    cSetInputTable
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionInput_cSetInputTable(JNIEnv * env, jobject thisObj,
                                                                                                                     jlong inputAddr, jint id,
                                                                                                                     jlong ntAddr)
{
    if (id != classifier::prediction::data) return;

    jniInput<logistic_regression::prediction::Input>::set<classifier::prediction::NumericTableInputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_prediction_PredictionInput
 * Method:    cSetInputModel
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionInput_cSetInputModel(JNIEnv * env, jobject thisObj,
                                                                                                                     jlong inputAddr, jint id,
                                                                                                                     jlong ntAddr)
{
    if (id != classifier::prediction::model) return;

    jniInput<logistic_regression::prediction::Input>::set<classifier::prediction::ModelInputId, logistic_regression::Model>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_prediction_PredictionInput
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionInput_cGetInputTable(JNIEnv * env, jobject thisObj,
                                                                                                                      jlong inputAddr, jint id)
{
    if (id != classifier::prediction::data) return (jlong)-1;

    return jniInput<logistic_regression::prediction::Input>::get<classifier::prediction::NumericTableInputId, NumericTable>(inputAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_prediction_PredictionInput
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionInput_cGetInputModel(JNIEnv * env, jobject thisObj,
                                                                                                                      jlong inputAddr, jint id)
{
    if (id != classifier::prediction::model) return (jlong)-1;

    return jniInput<logistic_regression::prediction::Input>::get<classifier::prediction::ModelInputId, logistic_regression::Model>(inputAddr, id);
}
