/* file: init_input.cpp */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
#include "com_intel_daal_algorithms_em_gmm_init_InitInput.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::em_gmm::init;

#define Data com_intel_daal_algorithms_em_gmm_init_InitInputId_Data

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitInput
 * Method:    cInit
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitInput_cInit(JNIEnv * env, jobject thisObj, jlong algAddr, jint prec,
                                                                                    jint method, jint cmode)
{
    return jniBatch<em_gmm::init::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitInput
 * Method:    cSetInput
 * Signature: (JIJ)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitInput_cSetInput(JNIEnv * env, jobject thisObj, jlong inputAddr, jint id,
                                                                                       jlong ntAddr)
{
    jniInput<em_gmm::init::Input>::set<em_gmm::init::InputId, NumericTable>(inputAddr, id, ntAddr);
    return (jint)0;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitInput
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitInput_cGetInputTable(JNIEnv * env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<em_gmm::init::Input>::get<em_gmm::init::InputId, NumericTable>(inputAddr, id);
}
