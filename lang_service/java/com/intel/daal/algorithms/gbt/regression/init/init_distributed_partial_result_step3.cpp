/* file: init_distributed_partial_result_step3.cpp */
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
#include "com_intel_daal_algorithms_gbt_regression_init_InitDistributedPartialResultStep3.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::gbt::regression::init;

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_init_InitDistributedPartialResultStep3_cNewInitDistributedPartialResultStep3
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<DistributedPartialResultStep3>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_gbt_regression_init_InitDistributedPartialResultStep3
 * Method:    cGetNumericTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_init_InitDistributedPartialResultStep3_cGetNumericTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<DistributedPartialResultStep3>::get<DistributedPartialResultStep3Id, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_gbt_regression_init_InitDistributedPartialResultStep3
 * Method:    cSetNumericTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_gbt_regression_init_InitDistributedPartialResultStep3_cSetNumericTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<DistributedPartialResultStep3>::set<DistributedPartialResultStep3Id, NumericTable>(resAddr, id, ntAddr);
}
