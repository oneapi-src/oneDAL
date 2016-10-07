/* file: training_distributed_partial_result_step4.cpp */
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

#include "implicit_als/training/JDistributedPartialResultStep4.h"

#include "implicit_als_training_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als;
using namespace daal::algorithms::implicit_als::training;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedPartialResultStep4
 * Method:    cNewDistributedPartialResultStep4
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedPartialResultStep4_cNewDistributedPartialResultStep4
  (JNIEnv *env, jobject thisObj)
{
    return jniArgument<DistributedPartialResultStep4>::newObj();
}


/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedPartialResultStep4
 * Method:    cGetDistributedPartialResultStep4
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedPartialResultStep4_cGetDistributedPartialResultStep4
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step4Local, implicit_als::training::Method, Distributed, fastCSR, defaultDense>::
        getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedPartialResultStep4
 * Method:    cGetPartialModel
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedPartialResultStep4_cGetPartialModel
  (JNIEnv *env, jobject thisObj, jlong partialResultAddr, jint id)
{
    return jniArgument<implicit_als::training::DistributedPartialResultStep4>::
        get<DistributedPartialResultStep4Id, PartialModel>(partialResultAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedPartialResultStep4
 * Method:    cSetPartialModel
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedPartialResultStep4_cSetPartialModel
  (JNIEnv *env, jobject thisObj, jlong partialResultAddr, jint id, jlong partialModelAddr)
{
    jniArgument<implicit_als::training::DistributedPartialResultStep4>::
        set<DistributedPartialResultStep4Id, PartialModel>(partialResultAddr, id, partialModelAddr);
}
