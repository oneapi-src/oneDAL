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
#include "qr_types.i"

#include "JComputeMode.h"
#include "qr/JInput.h"
#include "qr/JDistributedStep2MasterInput.h"
#include "qr/JDistributedStep3LocalInput.h"
#include "qr/JDistributedStep3LocalInputId.h"
#include "qr/JMethod.h"

#include "common_helpers.h"

#define inputOfStep3FromStep1Id com_intel_daal_algorithms_qr_DistributedStep3LocalInputId_inputOfStep3FromStep1Id
#define inputOfStep3FromStep2Id com_intel_daal_algorithms_qr_DistributedStep3LocalInputId_inputOfStep3FromStep2Id

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::qr;

/*
 * Class:     com_intel_daal_algorithms_qr_Input
 * Method:    cSetInputTable
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_qr_Input_cSetInputTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if(id != data) { return; }

    jniInput<qr::Input>::set<qr::InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_qr_Input
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_qr_Input_cGetInputTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if(id != data) { return (jlong) - 1; }

    return jniInput<qr::Input>::get<qr::InputId, NumericTable>(inputAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_qr_DistributedStep2MasterInput
 * Method:    cAddDataCollection
 * Signature:(JIIIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_qr_DistributedStep2MasterInput_cAddDataCollection
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint key, jlong dcAddr)
{
    jniInput<qr::DistributedStep2Input>::add<qr::MasterInputId, DataCollection>(inputAddr, qr::inputOfStep2FromStep1, key, dcAddr);
}

/*
 * Class:     com_intel_daal_algorithms_qr_DistributedStep3LocalInput
 * Method:    cSetDataCollection
 * Signature:(JIIIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_qr_DistributedStep3LocalInput_cSetDataCollection
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong dcAddr)
{
    if( id != inputOfStep3FromStep1 && id != inputOfStep3FromStep2 ) { return; }

    jniInput<qr::DistributedStep3Input>::set<qr::FinalizeOnLocalInputId, DataCollection>(inputAddr, id, dcAddr);
}
