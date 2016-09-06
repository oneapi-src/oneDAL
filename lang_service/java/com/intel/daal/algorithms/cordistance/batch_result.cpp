/* file: batch_result.cpp */
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_cordistance_Result */

#include "daal.h"
#include "cordistance/JResult.h"
#include "cordistance/JResultId.h"
#include "cordistance/JMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::correlation_distance;

#define DefaultMethodValue com_intel_daal_algorithms_cordistance_Method_DefaultMethodValue
#define DefaultResultId com_intel_daal_algorithms_cordistance_ResultId_DefaultResultId


JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cordistance_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<correlation_distance::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_cordistance_Result
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cordistance_Result_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<correlation_distance::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_cordistance_Result
 * Method:    cGetResultTable
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_cordistance_Result_cGetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<correlation_distance::Result>::get<correlation_distance::ResultId, NumericTable>(resAddr, id);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_cordistance_Result_cSetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<correlation_distance::Result>::set<correlation_distance::ResultId, NumericTable>(resAddr, id, ntAddr);
}
