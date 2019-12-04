/* file: partialresult.cpp */
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
#include "com_intel_daal_algorithms_covariance_PartialResult.h"

#include "covariance_types.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::covariance;

/*
 * Class:     com_intel_daal_algorithms_covariance_PartialResult
 * Method:    cNewPartialResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_PartialResult_cNewPartialResult(JNIEnv * env, jobject thisObj)
{
    return jniArgument<covariance::PartialResult>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_covariance_PartialResult
 * Method:    cGetPartialResultTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_PartialResult_cGetPartialResultTable(JNIEnv * env, jobject thisObj, jlong resAddr,
                                                                                                       jint id)
{
    return jniArgument<covariance::PartialResult>::get<covariance::PartialResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_covariance_PartialResult
 * Method:    cSetPartialResultTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_PartialResult_cSetPartialResultTable(JNIEnv * env, jobject thisObj, jlong resAddr,
                                                                                                      jint id, jlong ntAddr)
{
    jniArgument<covariance::PartialResult>::set<covariance::PartialResultId, NumericTable>(resAddr, id, ntAddr);
}
