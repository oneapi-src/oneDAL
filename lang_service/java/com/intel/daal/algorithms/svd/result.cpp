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

#include "JComputeMode.h"
#include "JComputeStep.h"
#include "svd/JMethod.h"
#include "svd/JResult.h"
#include "svd/JResultId.h"

#include "common_helpers.h"

#define singularValuesId com_intel_daal_algorithms_svd_ResultId_singularValuesId
#define leftSingularMatrixId com_intel_daal_algorithms_svd_ResultId_leftSingularMatrixId
#define rightSingularMatrixId com_intel_daal_algorithms_svd_ResultId_rightSingularMatrixId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::svd;

/*
 * Class:     Java_com_intel_daal_algorithms_qr_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<svd::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_svd_Result
 * Method:    cGetFactor
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_Result_cGetFactor
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if ( id == singularValuesId )
    {
        return jniArgument<svd::Result>::get<svd::ResultId, NumericTable>(resAddr, svd::singularValues);
    }
    else if(id == leftSingularMatrixId)
    {
        return jniArgument<svd::Result>::get<svd::ResultId, NumericTable>(resAddr, svd::leftSingularMatrix);
    }
    else if(id == rightSingularMatrixId)
    {
        return jniArgument<svd::Result>::get<svd::ResultId, NumericTable>(resAddr, svd::rightSingularMatrix);
    }

    return (jlong)0;
}
/*
 * Class:     com_intel_daal_algorithms_svd_Result
 * Method:    cGetResultTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_Result_cGetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<svd::Result>::get<svd::ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_svd_Result
 * Method:    cSetResultTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svd_Result_cSetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<svd::Result>::set<svd::ResultId, NumericTable>(resAddr, id, ntAddr);
}
