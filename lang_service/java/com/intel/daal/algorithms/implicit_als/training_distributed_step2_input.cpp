/* file: training_distributed_step2_input.cpp */
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

#include "implicit_als/training/JDistributedStep2MasterInput.h"

#include "implicit_als_training_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::training;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep2MasterInput
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep2MasterInput_cGetInput
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, algorithms::implicit_als::training::Method, Distributed, fastCSR>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep2MasterInput
 * Method:    cAddDataCollection
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep2MasterInput_cAddDataCollection
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong partialResultAddr)
{
    jniInput<DistributedInput<step2Master> >::add<MasterInputId, DistributedPartialResultStep1>(inputAddr, id, partialResultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep2MasterInput
 * Method:    cGetDataCollection
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep2MasterInput_cGetDataCollection
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<DistributedInput<step2Master> >::get<MasterInputId, DataCollection>(inputAddr, id);
}
