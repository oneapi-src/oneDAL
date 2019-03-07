/* file: input.cpp */
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
#include "pivoted_qr_types.i"

#include "JComputeMode.h"
#include "pivoted_qr/JInput.h"
#include "pivoted_qr/JMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::pivoted_qr;

/*
 * Class:     com_intel_daal_algorithms_pivoted_1qr_Input
 * Method:    cSetInputTable
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pivoted_1qr_Input_cSetInputTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if(id != data) return;
    jniInput<pivoted_qr::Input>::set<pivoted_qr::InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pivoted_1qr_Input
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pivoted_1qr_Input_cGetInputTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if(id != data) return (jlong)0;
    return jniInput<pivoted_qr::Input>::get<pivoted_qr::InputId, NumericTable>(inputAddr, id);
}
