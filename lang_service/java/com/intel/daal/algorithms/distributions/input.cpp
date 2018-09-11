/* file: input.cpp */
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
#include "distributions/JInput.h"
#include "distributions/JInputId.h"

#include "daal.h"

#include "common_helpers.h"

#define tableToFillId com_intel_daal_algorithms_distributions_InputId_tableToFillId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_distributions_Input
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_distributions_Input_cSetInput
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong NumericTableAddr)
{
    if (id == tableToFillId)
    {
        jniInput<distributions::Input>::set<distributions::InputId, NumericTable>(inputAddr, id, NumericTableAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_distributions_Input
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_distributions_Input_cGetInput
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<distributions::Input>::get<distributions::InputId, NumericTable>(inputAddr, id);
}
