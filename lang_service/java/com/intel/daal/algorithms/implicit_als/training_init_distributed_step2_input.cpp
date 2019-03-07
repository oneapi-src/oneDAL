/* file: training_init_distributed_step2_input.cpp */
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

#include "daal.h"

#include "implicit_als/training/init/JInitDistributedStep2LocalInput.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::training::init;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitDistributedStep2LocalInput
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedStep2LocalInput_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Local, implicit_als::training::init::Method, Distributed, fastCSR>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitDistributedStep2LocalInput
 * Method:    cSetDataCollection
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedStep2LocalInput_cSetDataCollection
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong collectionAddr)
{
    jniInput<DistributedInput<step2Local> >::
        set<Step2LocalInputId, KeyValueDataCollection>(inputAddr, id, collectionAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitDistributedStep2LocalInput
 * Method:    cGetDataCollection
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitDistributedStep2LocalInput_cGetDataCollection
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<DistributedInput<step2Local> >::
        get<Step2LocalInputId, KeyValueDataCollection>(inputAddr, id);
}
