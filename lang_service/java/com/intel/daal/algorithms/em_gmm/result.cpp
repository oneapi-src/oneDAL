/* file: result.cpp */
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

#include "daal.h"
#include "em_gmm/JResult.h"
#include "em_gmm/JResultId.h"
#include "em_gmm/JResultCovariancesId.h"
#include "em_gmm/JMethod.h"
#include "common_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::em_gmm;

#define DefaultDense    com_intel_daal_algorithms_em_gmm_Method_defaultDenseValue

#define WeightsValue      com_intel_daal_algorithms_em_gmm_ResultId_weightsValue
#define MeansValue        com_intel_daal_algorithms_em_gmm_ResultId_meansValue
#define CovariancesValue  com_intel_daal_algorithms_em_gmm_ResultCovariancesId_covariancesValue
#define NIterationsValue  com_intel_daal_algorithms_em_gmm_ResultId_nIterationsValue
#define GoalFunctionValue com_intel_daal_algorithms_em_gmm_ResultId_goalFunctionValue

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<em_gmm::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Result
 * Method:    cGetResultTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_Result_cGetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<em_gmm::Result>::get<em_gmm::ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Result
 * Method:    cGetCovariancesDataCollection
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_Result_cGetCovariancesDataCollection
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<em_gmm::Result>::get<em_gmm::ResultCovariancesId, DataCollection>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Result
 * Method:    cGetResultCovarianceTable
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_Result_cGetResultCovarianceTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jint index)
{
    return jniArgument<em_gmm::Result>::get<em_gmm::ResultCovariancesId, NumericTable>(resAddr, covariances, index);
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Result
 * Method:    cSetResultTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_Result_cSetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    ResultId cid;
    if(id == WeightsValue) { cid = weights; }
    else if(id == MeansValue) { cid = means; }
    else if(id == NIterationsValue) { cid = nIterations; }
    else if(id == GoalFunctionValue) { cid = goalFunction; }
    else { return; }

    jniArgument<em_gmm::Result>::set<em_gmm::ResultId, NumericTable>(resAddr, cid, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Result
 * Method:    sSetCovarianceCollection
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_Result_sSetCovarianceCollection
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong dcAddr)
{
    ResultCovariancesId cid;
    if(id == CovariancesValue) { cid = covariances; }
    else { return; }

    jniArgument<em_gmm::Result>::set<em_gmm::ResultCovariancesId, DataCollection>(resAddr, cid, dcAddr);
}
