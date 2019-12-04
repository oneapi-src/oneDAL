/* file: batch_result.cpp */
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
#include "com_intel_daal_algorithms_pca_transform_TransformResult.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::pca::transform;

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformResult_cNewResult(JNIEnv * env, jobject thisObj)
{
    return jniArgument<pca::transform::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformResult
 * Method:    cGetTransformedData
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformResult_cGetTransformedData(JNIEnv * env, jobject thisObj, jlong resAddr)
{
    return jniArgument<pca::transform::Result>::get<pca::transform::ResultId, NumericTable>(resAddr, pca::transform::transformedData);
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformResult
 * Method:    cSetTransformedData
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformResult_cSetTransformedData(JNIEnv * env, jobject thisObj, jlong resAddr,
                                                                                                        jlong ntAddr)
{
    jniArgument<pca::transform::Result>::set<pca::transform::ResultId, NumericTable>(resAddr, pca::transform::transformedData, ntAddr);
}
