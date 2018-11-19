/* file: result.cpp */
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

#include "JComputeMode.h"
#include "JComputeStep.h"
#include "qr/JMethod.h"
#include "qr/JResult.h"
#include "qr/JResultId.h"

#include "common_helpers.h"

#define matrixQId com_intel_daal_algorithms_qr_ResultId_matrixQId
#define matrixRId com_intel_daal_algorithms_qr_ResultId_matrixRId

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::qr;

/*
 * Class:     Java_com_intel_daal_algorithms_qr_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_qr_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<qr::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_qr_Result
 * Method:    cGetResultTable
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_qr_Result_cGetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if ( id == matrixQId )
    {
        return jniArgument<qr::Result>::get<qr::ResultId, NumericTable>(resAddr, qr::matrixQ);
    }
    else if(id == matrixRId)
    {
        return jniArgument<qr::Result>::get<qr::ResultId, NumericTable>(resAddr, qr::matrixR);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_qr_Result
 * Method:    cSetResultTable
 * Signature:(JI)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_qr_Result_cSetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<qr::Result>::set<qr::ResultId, NumericTable>(resAddr, id, ntAddr);
}
