/* file: partialresult.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#include <jni.h>
#include "JComputeMode.h"
#include "JComputeStep.h"
#include "svd/JOnlinePartialResult.h"
#include "svd/JDistributedStep2MasterPartialResult.h"
#include "svd/JPartialResultId.h"
#include "svd/JDistributedStep2MasterInputId.h"
#include "svd/JMethod.h"

#include "common_helpers.h"

#define outputOfStep1ForStep3Val com_intel_daal_algorithms_svd_PartialResultId_outputOfStep1ForStep3Id
#define outputOfStep1ForStep2Val com_intel_daal_algorithms_svd_PartialResultId_outputOfStep1ForStep2Id

#define inputOfStep2FromStep1Val com_intel_daal_algorithms_svd_DistributedStep2MasterInputId_inputOfStep2FromStep1Id

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::svd;

/*
 * Class:     com_intel_daal_algorithms_svd_PartialResult
 * Method:    cGetDataCollection
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_OnlinePartialResult_cGetDataCollection
(JNIEnv *env, jobject thisObj, jlong presAddr, jint id)
{
    if ( id == outputOfStep1ForStep3Val )
    {
        return jniArgument<svd::OnlinePartialResult>::get<svd::PartialResultId, DataCollection>(presAddr, svd::outputOfStep1ForStep3);
    }
    else if(id == outputOfStep1ForStep2Val)
    {
        return jniArgument<svd::OnlinePartialResult>::get<svd::PartialResultId, DataCollection>(presAddr, svd::outputOfStep1ForStep2);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep2MasterPartialResult
 * Method:    cGetKeyValueDataCollection
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_daal_algorithms_svd_DistributedStep2MasterPartialResult_cGetKeyValueDataCollection
(JNIEnv *env, jobject thisObj, jlong presAddr, jint id)
{
    if(id == inputOfStep2FromStep1Val)
    {
        return jniArgument<svd::DistributedPartialResult>::
            get<svd::DistributedPartialResultCollectionId, KeyValueDataCollection>(presAddr, svd::outputOfStep2ForStep3);
    }

    return (jlong)0;
}


/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep2MasterPartialResult
 * Method:    cGetResult
 * Signature:(JI)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svd_OnlinePartialResult_cSetOnlinePartialResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<svd::OnlinePartialResult>::set<svd::PartialResultId, DataCollection>(resAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep2MasterPartialResult
 * Method:    cGetResult
 * Signature:(JI)J
 */
JNIEXPORT void JNICALL
Java_com_intel_daal_algorithms_svd_DistributedStep2MasterPartialResult_cSetDistributedStep2MasterPartialResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<svd::DistributedPartialResult>::set<svd::DistributedPartialResultId, svd::Result>(resAddr, id, ntAddr);
}
