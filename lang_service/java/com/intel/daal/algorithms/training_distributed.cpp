/* file: training_distributed.cpp */
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
#include "JTrainingDistributed.h"

#include "daal_defines.h"
#include "algorithm.h"
#include "common_helpers_functions.h"

using namespace daal;
using namespace daal::services;
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_TrainingDistributed
 * Method:    cCompute
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_TrainingDistributed_cCompute
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Training<distributed> > alg =
        staticPointerCast<Training<distributed>, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->compute());
}


/*
 * Class:     com_intel_daal_algorithms_TrainingDistributed
 * Method:    cFinalizeCompute
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_TrainingDistributed_cFinalizeCompute
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Training<distributed> > alg =
        staticPointerCast<Training<distributed>, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->finalizeCompute());
}

/*
 * Class:     com_intel_daal_algorithms_TrainingDistributed
 * Method:    cCheckComputeParams
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_TrainingDistributed_cCheckComputeParams
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Training<distributed> > alg =
        staticPointerCast<Training<distributed>, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->checkComputeParams());
}


/*
 * Class:     com_intel_daal_algorithms_TrainingDistributed
 * Method:    cCheckFinalizeComputeParams
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_TrainingDistributed_cCheckFinalizeComputeParams
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Training<distributed> > alg =
        staticPointerCast<Training<distributed>, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->checkFinalizeComputeParams());
}

/*
 * Class:     com_intel_daal_algorithms_TrainingDistributed
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_TrainingDistributed_cDispose
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    delete(SharedPtr<AlgorithmIface> *)algAddr;
}
