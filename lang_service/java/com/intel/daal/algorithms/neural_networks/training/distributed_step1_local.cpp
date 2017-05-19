/* file: distributed_step1_local.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
#include "neural_networks/training/JDistributedStep1Local.h"

#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep1Local
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep1Local_cInit
  (JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniDistributed<step1Local, training::Method, training::Distributed, training::defaultDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep1Local
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep1Local_cInitParameter
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local, training::Method, training::Distributed, training::defaultDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep1Local
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep1Local_cGetInput
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local, training::Method, training::Distributed, training::defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep1Local
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep1Local_cGetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local, training::Method, training::Distributed, training::defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep1Local
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep1Local_cSetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resAddr)
{
    jniDistributed<step1Local, training::Method, training::Distributed, training::defaultDense>::setResult<training::Result>(prec, method, algAddr, resAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep1Local
 * Method:    cGetPartialResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep1Local_cGetPartialResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local, training::Method, training::Distributed, training::defaultDense>::getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep1Local
 * Method:    cSetPartialResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep1Local_cSetPartialResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resAddr)
{
    jniDistributed<step1Local, training::Method, training::Distributed, training::defaultDense>::setPartialResult<training::PartialResult>(prec, method, algAddr, resAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep1Local
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep1Local_cClone
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local, training::Method, training::Distributed, training::defaultDense>::getClone(prec, method, algAddr);
}
