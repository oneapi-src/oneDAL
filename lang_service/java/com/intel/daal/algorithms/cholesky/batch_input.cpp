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

#include <jni.h>
#include "cholesky/JInput.h"
#include "cholesky/JInputId.h"
#include "cholesky/JMethod.h"
#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::cholesky;

#define DefaultMethodValue com_intel_daal_algorithms_cholesky_Method_DefaultMethodValue
#define InputDataId        com_intel_daal_algorithms_cholesky_InputId_InputDataId

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cholesky_Input_cInit
(JNIEnv *jenv, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<cholesky::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_cholesky_Input_cSetInput
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<cholesky::Input>::set<cholesky::InputId, NumericTable>(inputAddr, id, ntAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cholesky_Input_cGetInputTable
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<cholesky::Input>::get<cholesky::InputId, NumericTable>(inputAddr, id);
}
