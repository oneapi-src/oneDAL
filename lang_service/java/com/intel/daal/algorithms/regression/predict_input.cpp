/* file: predict_input.cpp */
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
#include "com_intel_daal_algorithms_regression_prediction_PredictionInput.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::regression;
using namespace daal::algorithms::regression::prediction;

/*
 * Class:     com_intel_daal_algorithms_regression_prediction_PredictionInput
 * Method:    cInit
 * Signature: (JIJ)I
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_regression_prediction_PredictionInput_cInit(JNIEnv * env, jobject thisObj, jlong algAddr)
{
    regression::prediction::Input * inputPtr = NULL;

    SharedPtr<Batch> alg = staticPointerCast<Batch, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
    inputPtr             = alg->getInput();

    return (jlong)inputPtr;
}

/*
 * Class:     com_intel_daal_algorithms_regression_prediction_PredictionInput
 * Method:    cSetInputTable
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_regression_prediction_PredictionInput_cSetInputTable(JNIEnv * env, jobject thisObj,
                                                                                                           jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<regression::prediction::Input>::set<regression::prediction::NumericTableInputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_regression_prediction_PredictionInput
 * Method:    cSetInputModel
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_regression_prediction_PredictionInput_cSetInputModel(JNIEnv * env, jobject thisObj,
                                                                                                           jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<regression::prediction::Input>::set<regression::prediction::ModelInputId, regression::Model>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_regression_prediction_PredictionInput
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_regression_prediction_PredictionInput_cGetInputTable(JNIEnv * env, jobject thisObj,
                                                                                                            jlong inputAddr, jint id)
{
    return jniInput<regression::prediction::Input>::get<regression::prediction::NumericTableInputId, NumericTable>(inputAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_regression_prediction_PredictionInput
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_regression_prediction_PredictionInput_cGetInputModel(JNIEnv * env, jobject thisObj,
                                                                                                            jlong inputAddr, jint id)
{
    return jniInput<regression::prediction::Input>::get<regression::prediction::ModelInputId, regression::Model>(inputAddr, id);
}
