/* file: training_init_distributed_step2_input.cpp */
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

#include "daal.h"

#include "com_intel_daal_algorithms_implicit_als_training_init_InitDistributedStep2LocalInput.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::training::init;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitDistributedStep2LocalInput
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedStep2LocalInput_cGetInput(JNIEnv * env,
                                                                                                                            jobject thisObj,
                                                                                                                            jlong algAddr, jint prec,
                                                                                                                            jint method)
{
    return jniDistributed<step2Local, implicit_als::training::init::Method, Distributed, fastCSR>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitDistributedStep2LocalInput
 * Method:    cSetDataCollection
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedStep2LocalInput_cSetDataCollection(
    JNIEnv * env, jobject thisObj, jlong inputAddr, jint id, jlong collectionAddr)
{
    jniInput<DistributedInput<step2Local> >::set<Step2LocalInputId, KeyValueDataCollection>(inputAddr, id, collectionAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitDistributedStep2LocalInput
 * Method:    cGetDataCollection
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedStep2LocalInput_cGetDataCollection(JNIEnv * env,
                                                                                                                                     jobject thisObj,
                                                                                                                                     jlong inputAddr,
                                                                                                                                     jint id)
{
    return jniInput<DistributedInput<step2Local> >::get<Step2LocalInputId, KeyValueDataCollection>(inputAddr, id);
}
