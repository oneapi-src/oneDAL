/* file: training_init_distributed_partial_result_step2.cpp */
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

#include "com_intel_daal_algorithms_implicit_als_training_init_InitDistributedPartialResultStep2.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::training::init;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitDistributedPartialResultStep2
 * Method:    cNewPartialResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
    Java_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedPartialResultStep2_cNewPartialResult(JNIEnv * env, jobject thisObj)
{
    return jniArgument<DistributedPartialResultStep2>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitDistributedPartialResultStep2
 * Method:    cGetNumericTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedPartialResultStep2_cGetNumericTable(
    JNIEnv * env, jobject thisObj, jlong partialResultAddr, jint id)
{
    return jniArgument<DistributedPartialResultStep2>::get<DistributedPartialResultStep2Id, NumericTable>(partialResultAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitDistributedPartialResultStep2
 * Method:    cSetNumericTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedPartialResultStep2_cSetNumericTable(
    JNIEnv * env, jobject thisObj, jlong partialResultAddr, jint id, jlong numTableAddr)
{
    jniArgument<DistributedPartialResultStep2>::set<DistributedPartialResultStep2Id, NumericTable>(partialResultAddr, id, numTableAddr);
}
