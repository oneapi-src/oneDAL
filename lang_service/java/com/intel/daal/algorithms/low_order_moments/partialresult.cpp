/* file: partialresult.cpp */
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
#include "low_order_moments/JPartialResult.h"
#include "common_defines.i"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::low_order_moments;

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_PartialResult
 * Method:    cNewPartialResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_PartialResult_cNewPartialResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<low_order_moments::PartialResult>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_PartialResult
 * Method:    cGetPartialResultTable
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_PartialResult_cGetPartialResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<low_order_moments::PartialResult>::get<low_order_moments::PartialResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_PartialResult
 * Method:    cSetPartialResultTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_PartialResult_cSetPartialResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<low_order_moments::PartialResult>::set<low_order_moments::PartialResultId, NumericTable>(resAddr, id, ntAddr);
}
