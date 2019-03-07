/* file: batch_result.cpp */
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_cordistance_Result */

#include "daal.h"
#include "cordistance/JResult.h"
#include "cordistance/JResultId.h"
#include "cordistance/JMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::correlation_distance;

#define DefaultMethodValue com_intel_daal_algorithms_cordistance_Method_DefaultMethodValue
#define DefaultResultId com_intel_daal_algorithms_cordistance_ResultId_DefaultResultId


JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cordistance_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<correlation_distance::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_cordistance_Result
 * Method:    cGetResultTable
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cordistance_Result_cGetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<correlation_distance::Result>::get<correlation_distance::ResultId, NumericTable>(resAddr, id);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_cordistance_Result_cSetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<correlation_distance::Result>::set<correlation_distance::ResultId, NumericTable>(resAddr, id, ntAddr);
}
