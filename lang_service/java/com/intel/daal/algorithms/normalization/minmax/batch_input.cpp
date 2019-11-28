/* file: batch_input.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#include "com_intel_daal_algorithms_normalization_minmax_Input.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::normalization::minmax;

#include "com_intel_daal_algorithms_normalization_minmax_InputId.h"
#define InputDataId com_intel_daal_algorithms_normalization_minmax_InputId_InputDataId

/*
 * Class:     com_intel_daal_algorithms_normalization_minmax_Input
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_normalization_minmax_Input_cSetInputTable(JNIEnv * jenv, jobject thisObj, jlong inputAddr,
                                                                                                jint id, jlong ntAddr)
{
    if (id == InputDataId)
    {
        jniInput<normalization::minmax::Input>::set<normalization::minmax::InputId, NumericTable>(inputAddr, normalization::minmax::data, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_normalization_minmax_Input
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_normalization_minmax_Input_cGetInputTable(JNIEnv * jenv, jobject thisObj, jlong inputAddr,
                                                                                                 jint id)
{
    if (id == InputDataId)
    {
        return jniInput<normalization::minmax::Input>::get<normalization::minmax::InputId, NumericTable>(inputAddr, normalization::minmax::data);
    }

    return (jlong)0;
}
