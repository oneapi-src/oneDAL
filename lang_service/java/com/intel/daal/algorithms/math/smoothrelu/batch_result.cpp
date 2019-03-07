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

#include <jni.h>
#include "math/smoothrelu/JMethod.h"
#include "math/smoothrelu/JResult.h"
#include "math/smoothrelu/JResultId.h"
#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::math;

/*
 * Class:     com_intel_daal_algorithms_math_smoothrelu_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_math_smoothrelu_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<smoothrelu::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_math_smoothrelu_Result
 * Method:    cGetValue
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_math_smoothrelu_Result_cGetValue
(JNIEnv *env, jobject thisObj, jlong resAddr)
{
    return jniArgument<smoothrelu::Result>::get<smoothrelu::ResultId, NumericTable>(resAddr, smoothrelu::value);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_math_smoothrelu_Result_cSetValue
(JNIEnv *env, jobject thisObj, jlong resAddr, jlong ntAddr)
{
    jniArgument<smoothrelu::Result>::set<smoothrelu::ResultId, NumericTable>(resAddr, smoothrelu::value, ntAddr);
}
