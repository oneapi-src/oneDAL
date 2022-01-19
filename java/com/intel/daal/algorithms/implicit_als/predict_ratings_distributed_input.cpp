/* file: predict_ratings_distributed_input.cpp */
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

#include "com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsDistributedInput.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::implicit_als::prediction::ratings;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsDistributedInput
 * Method:    cInit
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_prediction_ratings_RatingsDistributedInput_cInit(JNIEnv * env, jobject thisObj,
                                                                                                                      jlong algAddr, jint prec,
                                                                                                                      jint method)
{
    return jniDistributed<step1Local, implicit_als::prediction::ratings::Method, Distributed, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsDistributedInput
 * Method:    cSetPartialModel
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_prediction_ratings_RatingsDistributedInput_cSetPartialModel(
    JNIEnv * env, jobject thisObj, jlong inputAddr, jint id, jlong modelAddr)
{
    jniInput<DistributedInput<step1Local> >::set<PartialModelInputId, algorithms::implicit_als::PartialModel>(inputAddr, id, modelAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_prediction_ratings_RatingsDistributedInput
 * Method:    cGetPartialModel
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_prediction_ratings_RatingsDistributedInput_cGetPartialModel(JNIEnv * env,
                                                                                                                                 jobject thisObj,
                                                                                                                                 jlong inputAddr,
                                                                                                                                 jint id)
{
    return jniInput<DistributedInput<step1Local> >::get<PartialModelInputId, algorithms::implicit_als::PartialModel>(inputAddr, id);
}
