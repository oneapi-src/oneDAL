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
#include "engines/JInput.h"
#include "engines/JInputId.h"

#include "daal.h"

#include "common_helpers.h"

#define tableToFill com_intel_daal_algorithms_engines_InputId_tableToFillId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_engines_Input
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_engines_Input_cSetInput
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong NumericTableAddr)
{
    if (id == tableToFill)
    {
        jniInput<engines::Input>::set<engines::InputId, NumericTable>(inputAddr, id, NumericTableAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_engines_Input
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_engines_Input_cGetInput
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<engines::Input>::get<engines::InputId, NumericTable>(inputAddr, id);
}
