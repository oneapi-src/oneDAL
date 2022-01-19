/* file: partialresultcorrelation.cpp */
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

#include "com_intel_daal_algorithms_pca_PartialCorrelationResult.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::pca;

/*
 * Class:     com_intel_daal_algorithms_pca_PartialCorrelationResult
 * Method:    cNewPartialCorrelationResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_PartialCorrelationResult_cNewPartialCorrelationResult(JNIEnv * env, jobject thisObj)
{
    return jniArgument<pca::PartialResult<correlationDense> >::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_pca_PartialCorrelationResult
 * Method:    cGetPartialCorrelationResultTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_PartialCorrelationResult_cGetPartialCorrelationResultTable(JNIEnv * env, jobject thisObj,
                                                                                                                      jlong resAddr, jint id)
{
    return jniArgument<pca::PartialResult<correlationDense> >::get<PartialCorrelationResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_pca_PartialCorrelationResult
 * Method:    cSetPartialCorrelationResultTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_PartialCorrelationResult_cSetPartialCorrelationResultTable(JNIEnv * env, jobject thisObj,
                                                                                                                     jlong resAddr, jint id,
                                                                                                                     jlong ntAddr)
{
    jniArgument<pca::PartialResult<correlationDense> >::set<PartialCorrelationResultId, NumericTable>(resAddr, id, ntAddr);
}
