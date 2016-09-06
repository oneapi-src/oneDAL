/* file: input.cpp */
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
#include "svd/JInput.h"
#include "svd/JDistributedStep2MasterInput.h"
#include "svd/JDistributedStep3LocalInput.h"
#include "svd/JDistributedStep3LocalInputId.h"
#include "svd/JMethod.h"

#include "common_defines.i"

#include "common_helpers.h"

#define inputOfStep3FromStep1Id com_intel_daal_algorithms_svd_DistributedStep3LocalInputId_inputOfStep3FromStep1Id
#define inputOfStep3FromStep2Id com_intel_daal_algorithms_svd_DistributedStep3LocalInputId_inputOfStep3FromStep2Id

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::svd;

/*
 * Class:     com_intel_daal_algorithms_svd_Input
 * Method:    cSetInputTable
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svd_Input_cSetInputTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if(id != data) { return; }

    jniInput<svd::Input>::set<InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_svd_Input
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_Input_cGetInputTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if(id != data) { return (jlong)-1; }

    return jniInput<svd::Input>::get<InputId, NumericTable>(inputAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep2MasterInput
 * Method:    cAddDataCollection
 * Signature:(JIIIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svd_DistributedStep2MasterInput_cAddDataCollection
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint key, jlong dcAddr)
{
    jniInput<svd::DistributedStep2Input>::add<MasterInputId, DataCollection>(inputAddr, svd::inputOfStep2FromStep1, key, dcAddr);
}

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep3LocalInput
 * Method:    cSetDataCollection
 * Signature:(JIIIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svd_DistributedStep3LocalInput_cSetDataCollection
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong dcAddr)
{
    if( id != inputOfStep3FromStep1 && id != inputOfStep3FromStep2 ) { return; }
    jniInput<svd::DistributedStep3Input>::set<FinalizeOnLocalInputId, DataCollection>(inputAddr, id, dcAddr);
}

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep2MasterInput
 * Method:    cAddDataCollection
 * Signature:(JIIIJ)I
 */
/*
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svd_DistributedStep2MasterInput_cAddDataCollection
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jint key, jlong dcAddr)
{
    using namespace daal::algorithms::svd;
    DataCollectionPtr shPtr =
        staticPointerCast<DataCollection, SerializationIface>(*(SerializationIfacePtr *)dcAddr);

    if ( prec == 0 )
    {
        SharedPtr<svd_dm2_d_def> alg = staticPointerCast<svd_dm2_d_def, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
        alg->input.add(svd::inputOfStep2FromStep1, key, shPtr);
    }
    else
    {
        SharedPtr<svd_dm2_s_def> alg = staticPointerCast<svd_dm2_s_def, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
        alg->input.add(svd::inputOfStep2FromStep1, key, shPtr);
    }
}
*/

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep3LocalInput
 * Method:    cSetDataCollection
 * Signature:(JIIIJ)I
 */
/*
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svd_DistributedStep3LocalInput_cSetDataCollection
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jint id, jlong dcAddr)
{
    using namespace daal::algorithms::svd;
    DataCollectionPtr shPtr =
        staticPointerCast<DataCollection, SerializationIface>(*(SerializationIfacePtr *)dcAddr);

    if ( prec == 0 )
    {
        if(id == inputOfStep3FromStep1Id)
        {
            SharedPtr<svd_dl3_d_def> alg = staticPointerCast<svd_dl3_d_def, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
            alg->input.set(svd::inputOfStep3FromStep1, shPtr);
        }
        else if(id == inputOfStep3FromStep2Id)
        {
            SharedPtr<svd_dl3_d_def> alg = staticPointerCast<svd_dl3_d_def, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
            alg->input.set(svd::inputOfStep3FromStep2, shPtr);
        }
    }
    else
    {
        if(id == inputOfStep3FromStep1Id)
        {
            SharedPtr<svd_dl3_s_def> alg = staticPointerCast<svd_dl3_s_def, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
            alg->input.set(svd::inputOfStep3FromStep1, shPtr);
        }
        else if(id == inputOfStep3FromStep2Id)
        {
            SharedPtr<svd_dl3_s_def> alg = staticPointerCast<svd_dl3_s_def, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
            alg->input.set(svd::inputOfStep3FromStep2, shPtr);
        }
    }
}
*/
