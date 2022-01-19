/* file: partialresult.cpp */
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
#include "com_intel_daal_algorithms_low_order_moments_PartialResult.h"
#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::low_order_moments;

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_PartialResult
 * Method:    cNewPartialResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_PartialResult_cNewPartialResult(JNIEnv * env, jobject thisObj)
{
    return jniArgument<low_order_moments::PartialResult>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_PartialResult
 * Method:    cGetPartialResultTable
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_PartialResult_cGetPartialResultTable(JNIEnv * env, jobject thisObj,
                                                                                                                jlong resAddr, jint id)
{
    return jniArgument<low_order_moments::PartialResult>::get<low_order_moments::PartialResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_PartialResult
 * Method:    cSetPartialResultTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_PartialResult_cSetPartialResultTable(JNIEnv * env, jobject thisObj,
                                                                                                               jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<low_order_moments::PartialResult>::set<low_order_moments::PartialResultId, NumericTable>(resAddr, id, ntAddr);
}
