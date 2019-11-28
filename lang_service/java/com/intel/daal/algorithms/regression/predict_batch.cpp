/* file: predict_batch.cpp */
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
#include "com_intel_daal_algorithms_regression_prediction_PredictionBatch.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::regression;
using namespace daal::algorithms::regression::prediction;
using namespace daal::services;

/*
 * Class:     com_intel_daal_algorithms_regression_prediction_PredictionBatch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_regression_prediction_PredictionBatch_cSetResult(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                       jlong resultAddr)
{
    SerializationIfacePtr * serializableShPtr = (SerializationIfacePtr *)resultAddr;
    regression::prediction::ResultPtr resultShPtr =
        services::staticPointerCast<regression::prediction::Result, SerializationIface>(*serializableShPtr);

    SharedPtr<Batch> alg = staticPointerCast<Batch, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
    alg->setResult(resultShPtr);
}
