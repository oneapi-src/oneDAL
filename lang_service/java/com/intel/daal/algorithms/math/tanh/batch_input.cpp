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
#include "math/tanh/JInput.h"
#include "math/tanh/JInputId.h"
#include "math/tanh/JMethod.h"
#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::math;

#define InputDataId        com_intel_daal_algorithms_math_tanh_InputId_dataId

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_math_tanh_Input_cSetInput
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if(id == InputDataId)
    {
        jniInput<tanh::Input>::set<tanh::InputId, NumericTable>(inputAddr, tanh::data, ntAddr);
    }
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_math_tanh_Input_cGetInputTable
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id)
{
    if(id == InputDataId)
    {
        return jniInput<tanh::Input>::get<tanh::InputId, NumericTable>(inputAddr, tanh::data);
    }

    return (jlong)0;
}
