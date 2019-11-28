/* file: predict_result.cpp */
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
#include "com_intel_daal_algorithms_regression_prediction_PredictionResult.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::regression;
using namespace daal::algorithms::regression::prediction;

/*
 * Class:     com_intel_daal_algorithms_regression_prediction_PredictionResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_regression_prediction_PredictionResult_cNewResult(JNIEnv * env, jobject thisObj)
{
    return jniArgument<regression::prediction::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_regression_prediction_PredictionResult
 * Method:    cGetResult
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_regression_prediction_PredictionResult_cGetResult(JNIEnv * env, jobject thisObj, jlong algAddr)
{
    SerializationIfacePtr * ptr = new SerializationIfacePtr();

    SharedPtr<Batch> alg = staticPointerCast<Batch, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
    *ptr                 = alg->getResult();
    return (jlong)ptr;
}
/*
 * Class:     com_intel_daal_algorithms_regression_prediction_PredictionResult
 * Method:    cGetMinimum
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_regression_prediction_PredictionResult_cGetResultTable(JNIEnv * env, jobject thisObj,
                                                                                                              jlong resAddr, jint id)
{
    return jniArgument<regression::prediction::Result>::get<regression::prediction::ResultId, NumericTable>(resAddr,
                                                                                                            (regression::prediction::ResultId)id);
}

/*
 * Class:     com_intel_daal_algorithms_regression_prediction_PredictionResult
 * Method:    cSetResultTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_regression_prediction_PredictionResult_cSetResultTable(JNIEnv * env, jobject thisObj,
                                                                                                             jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<regression::prediction::Result>::set<regression::prediction::ResultId, NumericTable>(resAddr, id, ntAddr);
}
