/* file: prediction_distributed.cpp */
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
#include "JPredictionDistributed.h"

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
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_PredictionDistributed_cCompute
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<DistributedPrediction> alg =
        staticPointerCast<DistributedPrediction, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->compute());
}

/*
 * Class:     com_intel_daal_algorithms_PredictionDistributed
 * Method:    cFinalizeCompute
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_PredictionDistributed_cFinalizeCompute
  (JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<DistributedPrediction> alg =
        staticPointerCast<DistributedPrediction, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->finalizeCompute());
}

/*
 * Class:     com_intel_daal_algorithms_PredictionDistributed
 * Method:    cCheckComputeParams
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_PredictionDistributed_cCheckComputeParams
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<DistributedPrediction> alg =
        staticPointerCast<DistributedPrediction, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->checkComputeParams());
}

/*
 * Class:     com_intel_daal_algorithms_PredictionDistributed
 * Method:    cCheckFinalizeComputeParams
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_PredictionDistributed_cCheckFinalizeComputeParams
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<DistributedPrediction> alg =
        staticPointerCast<DistributedPrediction, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->checkFinalizeComputeParams());
}

/*
 * Class:     com_intel_daal_algorithms_PredictionDistributed
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_PredictionDistributed_cDispose
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    delete(SharedPtr<AlgorithmIface> *)algAddr;
}
