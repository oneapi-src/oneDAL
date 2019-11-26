/* file: prediction_distributed.cpp */
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
#include "com_intel_daal_algorithms_PredictionDistributed.h"

#include "daal_defines.h"
#include "algorithm.h"
#include "common_helpers_functions.h"

using namespace daal::services;
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_PredictionDistributed
 * Method:    cCompute
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_PredictionDistributed_cCompute(JNIEnv * env, jobject thisObj, jlong algAddr)
{
    SharedPtr<DistributedPrediction> alg = staticPointerCast<DistributedPrediction, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->compute());
}

/*
 * Class:     com_intel_daal_algorithms_PredictionDistributed
 * Method:    cFinalizeCompute
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_PredictionDistributed_cFinalizeCompute(JNIEnv * env, jobject thisObj, jlong algAddr)
{
    SharedPtr<DistributedPrediction> alg = staticPointerCast<DistributedPrediction, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->finalizeCompute());
}

/*
 * Class:     com_intel_daal_algorithms_PredictionDistributed
 * Method:    cCheckComputeParams
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_PredictionDistributed_cCheckComputeParams(JNIEnv * env, jobject thisObj, jlong algAddr)
{
    SharedPtr<DistributedPrediction> alg = staticPointerCast<DistributedPrediction, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->checkComputeParams());
}

/*
 * Class:     com_intel_daal_algorithms_PredictionDistributed
 * Method:    cCheckFinalizeComputeParams
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_PredictionDistributed_cCheckFinalizeComputeParams(JNIEnv * env, jobject thisObj, jlong algAddr)
{
    SharedPtr<DistributedPrediction> alg = staticPointerCast<DistributedPrediction, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->checkFinalizeComputeParams());
}

/*
 * Class:     com_intel_daal_algorithms_PredictionDistributed
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_PredictionDistributed_cDispose(JNIEnv * env, jobject thisObj, jlong algAddr)
{
    delete (SharedPtr<AlgorithmIface> *)algAddr;
}
