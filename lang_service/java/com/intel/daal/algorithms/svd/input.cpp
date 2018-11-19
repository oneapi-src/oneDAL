/* file: input.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
