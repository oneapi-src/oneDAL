/* file: batch_result.cpp */
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
#include "com_intel_daal_algorithms_normalization_zscore_Result.h"

#include "daal.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::normalization::zscore;

/*
 * Class:     com_intel_daal_algorithms_normalization_zscore_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_normalization_zscore_Result_cNewResult(JNIEnv * env, jobject thisObj)
{
    return jniArgument<normalization::zscore::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_normalization_zscore_Result
 * Method:    cGetResultNumericTable
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_normalization_zscore_Result_cGetResultNumericTable(JNIEnv * env, jobject thisObj,
                                                                                                          jlong resAddr, jint id)
{
    return jniArgument<normalization::zscore::Result>::get<normalization::zscore::ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_normalization_zscore_Result
 * Method:    cSetResultNumericTable
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_normalization_zscore_Result_cSetResultNumericTable(JNIEnv * env, jobject thisObj, jlong resAddr,
                                                                                                         jint id, jlong ntAddr)
{
    jniArgument<normalization::zscore::Result>::set<normalization::zscore::ResultId, NumericTable>(resAddr, id, ntAddr);
}
