/* file: kernelfunction_input.cpp */
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
