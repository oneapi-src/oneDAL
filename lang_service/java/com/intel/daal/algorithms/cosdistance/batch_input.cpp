/* file: batch_input.cpp */
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_cosdistance_Input */

#include "daal.h"
#include "cosdistance/JInput.h"
#include "cosdistance/JMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::cosine_distance;

#define DefaultMethodValue com_intel_daal_algorithms_cosdistance_Method_DefaultMethodValue

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cosdistance_Input_cInit
(JNIEnv *jenv, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<cosine_distance::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_cosdistance_Input
 * Method:    cSetDataSet
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_cosdistance_Input_cSetInput
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<cosine_distance::Input>::set<cosine_distance::InputId, NumericTable>(inputAddr, id, ntAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cosdistance_Input_cGetInput
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<cosine_distance::Input>::get<cosine_distance::InputId, NumericTable>(inputAddr, id);
}
