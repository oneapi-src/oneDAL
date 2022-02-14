/* file: input.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <jni.h>
#include "com_intel_daal_algorithms_distributions_Input.h"

#include "daal.h"

#include "com/intel/daal/common_helpers.h"

#include "com_intel_daal_algorithms_distributions_InputId.h"
#define tableToFillId com_intel_daal_algorithms_distributions_InputId_tableToFillId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_distributions_Input
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_distributions_Input_cSetInput(JNIEnv * env, jobject thisObj, jlong inputAddr, jint id,
                                                                                    jlong NumericTableAddr)
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
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_distributions_Input_cGetInput(JNIEnv * env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<distributions::Input>::get<distributions::InputId, NumericTable>(inputAddr, id);
}
