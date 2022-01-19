/* file: predict_input.cpp */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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

#include "com_intel_daal_algorithms_linear_regression_prediction_Input.h"

#include "com/intel/daal/common_helpers.h"

#include "com_intel_daal_algorithms_linear_regression_prediction_PredictionMethod.h"
#define defaultDenseValue com_intel_daal_algorithms_linear_regression_prediction_PredictionMethod_defaultDenseValue
#include "com_intel_daal_algorithms_linear_regression_prediction_PredictionInputId.h"
#define dataId  com_intel_daal_algorithms_linear_regression_prediction_PredictionInputId_dataId
#define modelId com_intel_daal_algorithms_linear_regression_prediction_PredictionInputId_modelId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::linear_regression::prediction;

/*
 * Class:     com_intel_daal_algorithms_linear_regression_prediction_Input
 * Method:    cSetInput
 * Signature:(JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_linear_1regression_prediction_Input_cSetInput(JNIEnv * env, jobject thisObj, jlong inputAddr,
                                                                                                    jint id, jlong ntAddr)
{
    if (id == dataId)
    {
        jniInput<linear_regression::prediction::Input>::set<linear_regression::prediction::NumericTableInputId, NumericTable>(
            inputAddr, linear_regression::prediction::data, ntAddr);
    }
    else if (id == modelId)
    {
        jniInput<linear_regression::prediction::Input>::set<linear_regression::prediction::ModelInputId, linear_regression::Model>(
            inputAddr, linear_regression::prediction::model, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_prediction_PredictionBatch
 * Method:    cGetInput
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_prediction_Input_cGetInput(JNIEnv * env, jobject thisObj, jlong inputAddr,
                                                                                                     jint id)
{
    if (id == dataId)
    {
        return jniInput<linear_regression::prediction::Input>::get<linear_regression::prediction::NumericTableInputId, NumericTable>(
            inputAddr, linear_regression::prediction::data);
    }
    else if (id == modelId)
    {
        return jniInput<linear_regression::prediction::Input>::get<linear_regression::prediction::ModelInputId, linear_regression::Model>(
            inputAddr, linear_regression::prediction::model);
    }

    return (jlong)0;
}
