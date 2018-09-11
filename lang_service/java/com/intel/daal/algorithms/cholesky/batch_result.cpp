/* file: batch_result.cpp */
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
#include "cholesky/JMethod.h"
#include "cholesky/JResult.h"
#include "cholesky/JResultId.h"
#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::cholesky;

#define DefaultMethodValue com_intel_daal_algorithms_cholesky_Method_DefaultMethodValue
#define DefaultResultId com_intel_daal_algorithms_cholesky_ResultId_DefaultResultId

/*
 * Class:     com_intel_daal_algorithms_cholesky_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cholesky_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<cholesky::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_cholesky_Result
 * Method:    cGetCholeskyFactor
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cholesky_Result_cGetCholeskyFactor
(JNIEnv *env, jobject thisObj, jlong resAddr)
{
    return jniArgument<cholesky::Result>::get<cholesky::ResultId, NumericTable>(resAddr, choleskyFactor);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_cholesky_Result_cSetCholeskyFactor
(JNIEnv *env, jobject thisObj, jlong resAddr, jlong ntAddr)
{
    jniArgument<cholesky::Result>::set<cholesky::ResultId, NumericTable>(resAddr, choleskyFactor, ntAddr);
}
