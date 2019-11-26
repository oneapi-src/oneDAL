/* file: training_init_partial_result.cpp */
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

#include "com_intel_daal_algorithms_implicit_als_training_init_InitPartialResult.h"

#include "implicit_als_init_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::training::init;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitPartialResult
 * Method:    cNewPartialResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitPartialResult_cNewPartialResult(JNIEnv * env, jobject thisObj)
{
    return jniArgument<implicit_als::training::init::PartialResult>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitPartialResult
 * Method:    cGetPartialResultModel
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitPartialResult_cGetPartialResultModel(JNIEnv * env,
                                                                                                                            jobject thisObj,
                                                                                                                            jlong partialResultAddr,
                                                                                                                            jint id)
{
    return jniArgument<implicit_als::training::init::PartialResult>::get<PartialResultId, implicit_als::PartialModel>(partialResultAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitPartialResult
 * Method:    cSetPartialResultModel
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitPartialResult_cSetPartialResultModel(JNIEnv * env,
                                                                                                                           jobject thisObj,
                                                                                                                           jlong partialResultAddr,
                                                                                                                           jint id, jlong mdlAddr)
{
    jniArgument<implicit_als::training::init::PartialResult>::set<PartialResultId, implicit_als::PartialModel>(partialResultAddr, id, mdlAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitPartialResult
 * Method:    cGetPartialResultCollection
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitPartialResult_cGetPartialResultCollection(
    JNIEnv * env, jobject thisObj, jlong partialResultAddr, jint id)
{
    return jniArgument<implicit_als::training::init::PartialResult>::get<PartialResultCollectionId, KeyValueDataCollection>(partialResultAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitPartialResult
 * Method:    cSetPartialResultCollection
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitPartialResult_cSetPartialResultCollection(
    JNIEnv * env, jobject thisObj, jlong partialResultAddr, jint id, jlong collectionAddr)
{
    jniArgument<implicit_als::training::init::PartialResult>::set<PartialResultCollectionId, KeyValueDataCollection>(partialResultAddr, id,
                                                                                                                     collectionAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitPartialResult
 * Method:    cGetPartialResultTable
 * Signature: (JIJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitPartialResult_cGetPartialResultTable(JNIEnv * env,
                                                                                                                            jobject thisObj,
                                                                                                                            jlong partialResultAddr,
                                                                                                                            jint id, jlong key)
{
    return jniArgument<implicit_als::training::init::PartialResult>::get<PartialResultCollectionId, NumericTable>(partialResultAddr, id, key);
}
