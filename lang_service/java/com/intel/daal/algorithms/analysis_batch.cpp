/* file: analysis_batch.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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

#include "JAnalysisBatch.h"
#include "daal_defines.h"
#include "algorithm.h"
#include "daal.h"
#include "common_helpers_functions.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;

/*
 * Class:     com_intel_daal_algorithms_AnalysisBatch
 * Method:    cCompute
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_AnalysisBatch_cCompute
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Analysis<batch> > alg =
        staticPointerCast<Analysis<batch>, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->compute());
}

/*
 * Class:     com_intel_daal_algorithms_AnalysisBatch
 * Method:    cCheckComputeParams
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_AnalysisBatch_cCheckComputeParams
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Analysis<batch> > alg =
        staticPointerCast<Analysis<batch>, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->checkComputeParams());
}

/*
 * Class:     com_intel_daal_algorithms_AnalysisBatch
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_AnalysisBatch_cDispose
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    delete(SharedPtr<AlgorithmIface> *)algAddr;
}
