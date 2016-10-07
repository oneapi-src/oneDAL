/* file: kernelfunction_input.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#include "kernel_function/JInput.h"
#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kernel_1function_Input_cSetInput
(JNIEnv *env, jobject obj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<kernel_function::Input>::set<kernel_function::InputId, NumericTable>(inputAddr, id, ntAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kernel_1function_Input_cGetInput
(JNIEnv *env, jobject obj, jlong inputAddr, jint id)
{
    return jniInput<kernel_function::Input>::get<kernel_function::InputId, NumericTable>(inputAddr, id);
}
