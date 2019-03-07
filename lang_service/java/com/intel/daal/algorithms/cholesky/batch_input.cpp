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

#include <jni.h>
#include "cholesky/JInput.h"
#include "cholesky/JInputId.h"
#include "cholesky/JMethod.h"
#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::cholesky;

#define DefaultMethodValue com_intel_daal_algorithms_cholesky_Method_DefaultMethodValue
#define InputDataId        com_intel_daal_algorithms_cholesky_InputId_InputDataId

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cholesky_Input_cInit
(JNIEnv *jenv, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<cholesky::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_cholesky_Input_cSetInput
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<cholesky::Input>::set<cholesky::InputId, NumericTable>(inputAddr, id, ntAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cholesky_Input_cGetInputTable
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<cholesky::Input>::get<cholesky::InputId, NumericTable>(inputAddr, id);
}
