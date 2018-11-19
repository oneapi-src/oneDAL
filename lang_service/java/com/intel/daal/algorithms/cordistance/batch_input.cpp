/* file: batch_input.cpp */
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_cordistance_Input */

#include "daal.h"
#include "cordistance/JInput.h"
#include "cordistance/JMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::correlation_distance;

#define DefaultMethodValue com_intel_daal_algorithms_cordistance_Method_DefaultMethodValue


JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cordistance_Input_cInit
(JNIEnv *jenv, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<correlation_distance::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_cordistance_Input
 * Method:    cSetDataSet
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_cordistance_Input_cSetInput
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<correlation_distance::Input>::set<correlation_distance::InputId, NumericTable>(inputAddr, id, ntAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cordistance_Input_cGetInput
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<correlation_distance::Input>::get<correlation_distance::InputId, NumericTable>(inputAddr, id);
}
