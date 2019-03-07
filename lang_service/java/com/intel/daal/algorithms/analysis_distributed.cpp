/* file: analysis_distributed.cpp */
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

#include "JAnalysisDistributed.h"
#include "daal_defines.h"
#include "algorithm.h"
#include "common_helpers_functions.h"

using namespace daal;
using namespace daal::services;
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_AnalysisOffline
 * Method:    cCompute
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_AnalysisDistributed_cCompute
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Analysis<distributed> > alg =
        staticPointerCast<Analysis<distributed>, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->compute());
}


/*
 * Class:     com_intel_daal_algorithms_AnalysisDistributed
 * Method:    cFinalizeCompute
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_AnalysisDistributed_cFinalizeCompute
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Analysis<distributed> > alg =
        staticPointerCast<Analysis<distributed>, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->finalizeCompute());
}

/*
 * Class:     com_intel_daal_algorithms_AnalysisDistributed
 * Method:    cCheckFinalizeComputeParams
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_AnalysisDistributed_cCheckFinalizeComputeParams
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Analysis<distributed> > alg =
        staticPointerCast<Analysis<distributed>, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->checkFinalizeComputeParams());
}

/*
 * Class:     com_intel_daal_algorithms_AnalysisDistributed
 * Method:    cCheckComputeParams
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_AnalysisDistributed_cCheckComputeParams
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Analysis<distributed> > alg =
        staticPointerCast<Analysis<distributed>, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->checkComputeParams());
}

/*
 * Class:     com_intel_daal_algorithms_AnalysisDistributed
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_AnalysisDistributed_cDispose
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    delete(SharedPtr<AlgorithmIface> *)algAddr;
}
