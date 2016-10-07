/* file: partialresult.cpp */
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
#include "JComputeMode.h"
#include "JComputeStep.h"
#include "qr/JOnlinePartialResult.h"
#include "qr/JDistributedStep2MasterPartialResult.h"
#include "qr/JPartialResultId.h"
#include "qr/JDistributedStep2MasterInputId.h"
#include "qr/JMethod.h"

#include "qr_types.i"

#include "common_helpers.h"

#define outputOfStep1ForStep3Val com_intel_daal_algorithms_qr_PartialResultId_outputOfStep1ForStep3Id
#define outputOfStep1ForStep2Val com_intel_daal_algorithms_qr_PartialResultId_outputOfStep1ForStep2Id

#define inputOfStep2FromStep1Val com_intel_daal_algorithms_qr_DistributedStep2MasterInputId_inputOfStep2FromStep1Id

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::qr;

/*
 * Class:     com_intel_daal_algorithms_qr_OnlinePartialResult
 * Method:    cGetDataCollection
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_qr_OnlinePartialResult_cGetDataCollection
(JNIEnv *env, jobject thisObj, jlong presAddr, jint id)
{
    if ( id == outputOfStep1ForStep3Val )
    {
        return jniArgument<qr::OnlinePartialResult>::get<qr::PartialResultId, DataCollection>(presAddr, qr::outputOfStep1ForStep3);
    }
    else if(id == outputOfStep1ForStep2Val)
    {
        return jniArgument<qr::OnlinePartialResult>::get<qr::PartialResultId, DataCollection>(presAddr, qr::outputOfStep1ForStep2);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_qr_DistributedStep2MasterPartialResult
 * Method:    cGetKeyValueDataCollection
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_qr_DistributedStep2MasterPartialResult_cGetKeyValueDataCollection
(JNIEnv *env, jobject thisObj, jlong presAddr, jint id)
{
    if(id == inputOfStep2FromStep1Val)
    {
        return jniArgument<qr::DistributedPartialResult>::
            get<qr::DistributedPartialResultCollectionId, KeyValueDataCollection>(presAddr, qr::outputOfStep2ForStep3);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_qr_DistributedStep2MasterPartialResult
 * Method:    cGetResult
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_qr_DistributedStep2MasterPartialResult_cGetResult
(JNIEnv *env, jobject thisObj, jlong presAddr, jint id)
{
    if(id == inputOfStep2FromStep1Val)
    {
        return jniArgument<qr::DistributedPartialResult>::get<qr::DistributedPartialResultId, qr::Result>(presAddr, qr::finalResultFromStep2Master);
    }

    return (jlong)0;
}
