/* file: training_init_distributed_step2.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "com_intel_daal_algorithms_implicit_als_training_init_InitDistributedStep2Local.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::training::init;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitDistributedStep2Local
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedStep2Local_cInit(JNIEnv * env, jobject thisObj,
                                                                                                                   jint prec, jint method)
{
    return jniDistributed<step2Local, implicit_als::training::init::Method, Distributed, fastCSR>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitDistributedStep2Local
 * Method:    cGetPartialResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedStep2Local_cGetPartialResult(JNIEnv * env,
                                                                                                                               jobject thisObj,
                                                                                                                               jlong algAddr,
                                                                                                                               jint prec, jint method)
{
    return jniDistributed<step2Local, implicit_als::training::init::Method, Distributed, fastCSR>::getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitDistributedStep2Local
 * Method:    cSetPartialResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedStep2Local_cSetPartialResult(
    JNIEnv * env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniDistributed<step2Local, implicit_als::training::init::Method, Distributed, fastCSR>::setPartialResult<DistributedPartialResultStep2>(
        prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitDistributedStep2Local
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedStep2Local_cClone(JNIEnv * env, jobject thisObj,
                                                                                                                    jlong algAddr, jint prec,
                                                                                                                    jint method)
{
    return jniDistributed<step2Local, implicit_als::training::init::Method, Distributed, fastCSR>::getClone(prec, method, algAddr);
}
