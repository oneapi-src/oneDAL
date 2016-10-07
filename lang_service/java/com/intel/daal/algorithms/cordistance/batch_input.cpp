/* file: batch_input.cpp */
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_cordistance_Input */

#include "daal.h"
#include "cordistance/JInput.h"
#include "cordistance/JMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::correlation_distance;

#define DefaultMethodValue com_intel_daal_algorithms_cordistance_Method_DefaultMethodValue


JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cordistance_Input_cInit
(JNIEnv *jenv, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<correlation_distance::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_cordistance_Input
 * Method:    cSetDataSet
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_cordistance_Input_cSetInput
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<correlation_distance::Input>::set<correlation_distance::InputId, NumericTable>(inputAddr, id, ntAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cordistance_Input_cGetInput
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<correlation_distance::Input>::get<correlation_distance::InputId, NumericTable>(inputAddr, id);
}
