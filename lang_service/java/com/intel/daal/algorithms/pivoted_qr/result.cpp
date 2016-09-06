/* file: result.cpp */
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
#include "pivoted_qr_types.i"

#include "JComputeMode.h"
#include "JComputeStep.h"
#include "pivoted_qr/JMethod.h"
#include "pivoted_qr/JResult.h"
#include "pivoted_qr/JResultId.h"

#include "common_helpers.h"

#define matrixQId com_intel_daal_algorithms_pivoted_qr_ResultId_matrixQId
#define matrixRId com_intel_daal_algorithms_pivoted_qr_ResultId_matrixRId
#define permutationMatrixId com_intel_daal_algorithms_pivoted_qr_ResultId_permutationMatrixId

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::pivoted_qr;

/*
 * Class:     Java_com_intel_daal_algorithms_pivoted_1qr_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pivoted_1qr_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<pivoted_qr::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_pivoted_1qr_Result
 * Method:    cGetResultTable
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pivoted_1qr_Result_cGetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if ( id == matrixQId )
    {
        return jniArgument<pivoted_qr::Result>::get<pivoted_qr::ResultId, NumericTable>(resAddr, pivoted_qr::matrixQ);
    }
    else if(id == matrixRId)
    {
        return jniArgument<pivoted_qr::Result>::get<pivoted_qr::ResultId, NumericTable>(resAddr, pivoted_qr::matrixR);
    }
    else if(id == permutationMatrixId)
    {
        return jniArgument<pivoted_qr::Result>::get<pivoted_qr::ResultId, NumericTable>(resAddr, pivoted_qr::permutationMatrix);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_pivoted_1qr_Result
 * Method:    cSetResultTable
 * Signature:(JI)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pivoted_1qr_Result_cSetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<pivoted_qr::Result>::set<pivoted_qr::ResultId, NumericTable>(resAddr, id, ntAddr);
}
