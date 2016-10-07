/* file: distributedmasterinput.cpp */
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

#include "daal.h"
#include "covariance/JDistributedStep2MasterInput.h"

#include "covariance_types.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::covariance;

/*
 * Class:     com_intel_daal_algorithms_covariance_DistributedStep2MasterInput
 * Method:    cInit
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_DistributedStep2MasterInput_cInit
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, covariance::Method, Distributed, defaultDense, singlePassDense, sumDense,
        fastCSR, singlePassCSR, sumCSR>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_covariance_DistributedStep2MasterInput
 * Method:    cAddInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_DistributedStep2MasterInput_cAddInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong partialResultAddr)
{
    jniInput<covariance::DistributedInput<step2Master> >::add<covariance::MasterInputId, covariance::PartialResult>(inputAddr, id, partialResultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_covariance_Input
 * Method:    cSetCInputObject
 * Signature: (JJIIII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_DistributedStep2MasterInput_cSetCInputObject
(JNIEnv *env, jobject thisObj, jlong inputAddr, jlong algAddr, jint prec, jint method)
// somehow this function isn't called if has >4 parameters
{
    jniDistributed<step2Master, covariance::Method, Distributed, defaultDense, singlePassDense, sumDense,
        fastCSR, singlePassCSR, sumCSR>::setInput<covariance::DistributedInput<step2Master> >(prec, method, algAddr, inputAddr);
}
