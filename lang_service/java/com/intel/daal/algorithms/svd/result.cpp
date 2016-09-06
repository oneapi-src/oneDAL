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
#include "svd_types.i"

#include "JComputeMode.h"
#include "JComputeStep.h"
#include "svd/JMethod.h"
#include "svd/JResult.h"
#include "svd/JResultId.h"

#include "common_helpers.h"

#define singularValuesId com_intel_daal_algorithms_svd_ResultId_singularValuesId
#define leftSingularMatrixId com_intel_daal_algorithms_svd_ResultId_leftSingularMatrixId
#define rightSingularMatrixId com_intel_daal_algorithms_svd_ResultId_rightSingularMatrixId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::svd;

/*
 * Class:     Java_com_intel_daal_algorithms_qr_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<svd::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_svd_Result
 * Method:    cGetFactor
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_Result_cGetFactor
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if ( id == singularValuesId )
    {
        return jniArgument<svd::Result>::get<svd::ResultId, NumericTable>(resAddr, svd::singularValues);
    }
    else if(id == leftSingularMatrixId)
    {
        return jniArgument<svd::Result>::get<svd::ResultId, NumericTable>(resAddr, svd::leftSingularMatrix);
    }
    else if(id == rightSingularMatrixId)
    {
        return jniArgument<svd::Result>::get<svd::ResultId, NumericTable>(resAddr, svd::rightSingularMatrix);
    }

    return (jlong)0;
}
/*
 * Class:     com_intel_daal_algorithms_svd_Result
 * Method:    cGetResultTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_Result_cGetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<svd::Result>::get<svd::ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_svd_Result
 * Method:    cSetResultTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svd_Result_cSetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<svd::Result>::set<svd::ResultId, NumericTable>(resAddr, id, ntAddr);
}
