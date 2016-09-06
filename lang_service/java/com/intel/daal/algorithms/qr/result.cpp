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
#include "qr_types.i"

#include "JComputeMode.h"
#include "JComputeStep.h"
#include "qr/JMethod.h"
#include "qr/JResult.h"
#include "qr/JResultId.h"

#include "common_helpers.h"

#define matrixQId com_intel_daal_algorithms_qr_ResultId_matrixQId
#define matrixRId com_intel_daal_algorithms_qr_ResultId_matrixRId

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::qr;

/*
 * Class:     Java_com_intel_daal_algorithms_qr_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_qr_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<qr::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_qr_Result
 * Method:    cGetResultTable
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_qr_Result_cGetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if ( id == matrixQId )
    {
        return jniArgument<qr::Result>::get<qr::ResultId, NumericTable>(resAddr, qr::matrixQ);
    }
    else if(id == matrixRId)
    {
        return jniArgument<qr::Result>::get<qr::ResultId, NumericTable>(resAddr, qr::matrixR);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_qr_Result
 * Method:    cSetResultTable
 * Signature:(JI)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_qr_Result_cSetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<qr::Result>::set<qr::ResultId, NumericTable>(resAddr, id, ntAddr);
}
