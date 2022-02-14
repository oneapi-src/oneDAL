/* file: training_distributed_step3_input.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "daal.h"

#include "com_intel_daal_algorithms_implicit_als_training_DistributedStep3LocalInput.h"

#include "implicit_als_training_defines.i"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::training;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep3LocalInput
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep3LocalInput_cGetInput(JNIEnv * env, jobject thisObj,
                                                                                                                   jlong algAddr, jint prec,
                                                                                                                   jint method)
{
    return jniDistributed<step3Local, algorithms::implicit_als::training::Method, Distributed, fastCSR>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep3LocalInput
 * Method:    cSetPartialModel
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep3LocalInput_cSetPartialModel(JNIEnv * env,
                                                                                                                         jobject thisObj,
                                                                                                                         jlong inputAddr, jint id,
                                                                                                                         jlong partialModelAddr)
{
    jniInput<DistributedInput<step3Local> >::set<PartialModelInputId, algorithms::implicit_als::PartialModel>(inputAddr, id, partialModelAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep3LocalInput
 * Method:    cGetPartialModel
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep3LocalInput_cGetPartialModel(JNIEnv * env,
                                                                                                                          jobject thisObj,
                                                                                                                          jlong inputAddr, jint id)
{
    return jniInput<DistributedInput<step3Local> >::get<PartialModelInputId, algorithms::implicit_als::PartialModel>(inputAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep3LocalInput
 * Method:    cSetNumericTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep3LocalInput_cSetNumericTable(JNIEnv * env,
                                                                                                                         jobject thisObj,
                                                                                                                         jlong inputAddr, jint id,
                                                                                                                         jlong numTableAddr)
{
    jniInput<DistributedInput<step3Local> >::set<Step3LocalNumericTableInputId, NumericTable>(inputAddr, id, numTableAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep3LocalInput
 * Method:    cGetNumericTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep3LocalInput_cGetNumericTable(JNIEnv * env,
                                                                                                                          jobject thisObj,
                                                                                                                          jlong inputAddr, jint id)
{
    return jniInput<DistributedInput<step3Local> >::get<Step3LocalNumericTableInputId, NumericTable>(inputAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep3LocalInput
 * Method:    cSetDataCollection
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep3LocalInput_cSetDataCollection(JNIEnv * env,
                                                                                                                           jobject thisObj,
                                                                                                                           jlong inputAddr, jint id,
                                                                                                                           jlong collectionAddr)
{
    jniInput<DistributedInput<step3Local> >::set<Step3LocalCollectionInputId, KeyValueDataCollection>(inputAddr, id, collectionAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep3LocalInput
 * Method:    cGetDataCollection
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep3LocalInput_cGetDataCollection(JNIEnv * env,
                                                                                                                            jobject thisObj,
                                                                                                                            jlong inputAddr, jint id)
{
    return jniInput<DistributedInput<step3Local> >::get<Step3LocalCollectionInputId, KeyValueDataCollection>(inputAddr, id);
}
