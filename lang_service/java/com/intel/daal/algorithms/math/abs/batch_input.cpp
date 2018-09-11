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

#include <jni.h>
#include "math/abs/JInput.h"
#include "math/abs/JInputId.h"
#include "math/abs/JMethod.h"
#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::math;

#define FastCSRMethodValue com_intel_daal_algorithms_math_abs_Method_FastCSRMethodValue
#define InputDataId        com_intel_daal_algorithms_math_abs_InputId_dataId

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_math_abs_Input_cSetInput
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if(id == InputDataId)
    {
        jniInput<abs::Input>::set<abs::InputId, NumericTable>(inputAddr, abs::data, ntAddr);
    }
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_math_abs_Input_cGetInputTable
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id)
{
    if(id == InputDataId)
    {
        return jniInput<abs::Input>::get<abs::InputId, NumericTable>(inputAddr, abs::data);
    }

    return (jlong)0;
}
