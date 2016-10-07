/* file: training_init_input.cpp */
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

#include "daal.h"

#include "implicit_als/training/init/JInitInput.h"

#include "implicit_als_init_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::training::init;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitInput
 * Method:    cInit
 * Signature: (JIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitInput_cInit
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jint cmode)
{
    if (cmode == jBatch)
    {
        return jniBatch<implicit_als::training::init::Method, Batch, fastCSR, defaultDense>::getInput(prec, method, algAddr);
    }
    else if (cmode == jDistributed)
    {
        return jniDistributed<step1Local, implicit_als::training::init::Method, Distributed, fastCSR, defaultDense>::getInput(prec, method, algAddr);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitInput_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<implicit_als::training::init::Input>::set<InputId, NumericTable>(inputAddr, data, ntAddr);
}


/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitInput
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitInput_cGetInputTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)

{
    return jniInput<implicit_als::training::init::Input>::get<InputId, NumericTable>(inputAddr, data);
}
