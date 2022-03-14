/* file: result.cpp */
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_univariate_outlier_detection_Result */

#include "daal.h"
#include "com_intel_daal_algorithms_univariate_outlier_detection_Result.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::univariate_outlier_detection;

/*
 * Class:     com_intel_daal_algorithms_univariate_outlier_detection_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_univariate_1outlier_1detection_Result_cNewResult(JNIEnv * env, jobject thisObj)
{
    return jniArgument<univariate_outlier_detection::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_univariate_outlier_detection_Result
 * Method:    cGetResultTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_univariate_1outlier_1detection_Result_cGetResultTable(JNIEnv * env, jobject thisObj,
                                                                                                             jlong resAddr, jint id)
{
    return jniArgument<univariate_outlier_detection::Result>::get<univariate_outlier_detection::ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_univariate_outlier_detection_Result
 * Method:    cSetResultTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_univariate_1outlier_1detection_Result_cSetResultTable(JNIEnv * env, jobject thisObj,
                                                                                                            jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<univariate_outlier_detection::Result>::set<univariate_outlier_detection::ResultId, NumericTable>(resAddr, id, ntAddr);
}
