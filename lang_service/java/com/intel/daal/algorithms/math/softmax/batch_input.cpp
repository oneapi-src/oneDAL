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
#include "math/softmax/JInput.h"
#include "math/softmax/JInputId.h"
#include "math/softmax/JMethod.h"
#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::math;

#define InputDataId        com_intel_daal_algorithms_math_softmax_InputId_dataId

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_math_softmax_Input_cSetInput
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if(id == InputDataId)
    {
        jniInput<softmax::Input>::set<softmax::InputId, NumericTable>(inputAddr, softmax::data, ntAddr);
    }
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_math_softmax_Input_cGetInputTable
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id)
{
    if(id == InputDataId)
    {
        return jniInput<softmax::Input>::get<softmax::InputId, NumericTable>(inputAddr, softmax::data);
    }

    return (jlong)0;
}
