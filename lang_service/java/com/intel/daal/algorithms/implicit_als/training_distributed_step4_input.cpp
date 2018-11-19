/* file: training_distributed_step4_input.cpp */
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

#include "daal.h"

#include "implicit_als/training/JDistributedStep4LocalInput.h"

#include "implicit_als_training_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::training;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep4LocalInput
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep4LocalInput_cGetInput
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step4Local, algorithms::implicit_als::training::Method, Distributed, fastCSR>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep4LocalInput
 * Method:    cSetNumericTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep4LocalInput_cSetNumericTable
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong numTableAddr)
{
    jniInput<DistributedInput<step4Local> >::set<Step4LocalNumericTableInputId, NumericTable>(inputAddr, id, numTableAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep4LocalInput
 * Method:    cGetNumericTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep4LocalInput_cGetNumericTable
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<DistributedInput<step4Local> >::get<Step4LocalNumericTableInputId, NumericTable>(inputAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep4LocalInput
 * Method:    cSetDataCollection
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep4LocalInput_cSetDataCollection
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong collectionAddr)
{
    jniInput<DistributedInput<step4Local> >::set<Step4LocalPartialModelsInputId, KeyValueDataCollection>(inputAddr, id, collectionAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep4LocalInput
 * Method:    cGetDataCollection
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep4LocalInput_cGetDataCollection
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<DistributedInput<step4Local> >::get<Step4LocalPartialModelsInputId, KeyValueDataCollection>(inputAddr, id);
}
