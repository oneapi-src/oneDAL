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
#include "math/abs/JMethod.h"
#include "math/abs/JResult.h"
#include "math/abs/JResultId.h"
#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::math;

/*
 * Class:     com_intel_daal_algorithms_math_abs_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_math_abs_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<abs::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_math_abs_Result
 * Method:    cGetValue
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_math_abs_Result_cGetValue
(JNIEnv *env, jobject thisObj, jlong resAddr)
{
    return jniArgument<abs::Result>::get<abs::ResultId, NumericTable>(resAddr, abs::value);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_math_abs_Result_cSetValue
(JNIEnv *env, jobject thisObj, jlong resAddr, jlong ntAddr)
{
    jniArgument<abs::Result>::set<abs::ResultId, NumericTable>(resAddr, abs::value, ntAddr);
}
