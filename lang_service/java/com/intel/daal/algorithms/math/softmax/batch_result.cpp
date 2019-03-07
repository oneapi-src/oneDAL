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
#include "math/softmax/JMethod.h"
#include "math/softmax/JResult.h"
#include "math/softmax/JResultId.h"
#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::math;

/*
 * Class:     com_intel_daal_algorithms_softmax_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_math_softmax_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<softmax::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_softmax_Result
 * Method:    cGetsoftmaxFactor
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_math_softmax_Result_cGetValue
(JNIEnv *env, jobject thisObj, jlong resAddr)
{
    return jniArgument<softmax::Result>::get<softmax::ResultId, NumericTable>(resAddr, softmax::value);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_math_softmax_Result_cSetValue
(JNIEnv *env, jobject thisObj, jlong resAddr, jlong ntAddr)
{
    jniArgument<softmax::Result>::set<softmax::ResultId, NumericTable>(resAddr, softmax::value, ntAddr);
}
