/* file: distributedmaster.cpp */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
#include "com_intel_daal_algorithms_low_order_moments_DistributedStep2Master.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::low_order_moments;

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_Distributed
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_DistributedStep2Master_cInit(JNIEnv * env, jobject thisObj, jint prec,
                                                                                                        jint method)
{
    return jniDistributed<step2Master, low_order_moments::Method, Distributed, defaultDense, singlePassDense, sumDense, fastCSR, singlePassCSR,
                          sumCSR>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_DistributedStep2Master
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_DistributedStep2Master_cSetResult(JNIEnv * env, jobject thisObj,
                                                                                                            jlong algAddr, jint prec, jint method,
                                                                                                            jlong resultAddr)
{
    jniDistributed<step2Master, low_order_moments::Method, Distributed, defaultDense, singlePassDense, sumDense, fastCSR, singlePassCSR,
                   sumCSR>::setResult<low_order_moments::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_DistributedStep2Master
 * Method:    cSetPartialResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_DistributedStep2Master_cSetPartialResult(
    JNIEnv * env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong partialResultAddr, jboolean initFlag)
{
    jniDistributed<step2Master, low_order_moments::Method, Distributed, defaultDense, singlePassDense, sumDense, fastCSR, singlePassCSR,
                   sumCSR>::setPartialResult<low_order_moments::PartialResult>(prec, method, algAddr, partialResultAddr, initFlag);
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_DistributedStep2Master
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_DistributedStep2Master_cGetResult(JNIEnv * env, jobject thisObj,
                                                                                                             jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, low_order_moments::Method, Distributed, defaultDense, singlePassDense, sumDense, fastCSR, singlePassCSR,
                          sumCSR>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_DistributedStep2Master
 * Method:    cGetPartialResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_DistributedStep2Master_cGetPartialResult(JNIEnv * env, jobject thisObj,
                                                                                                                    jlong algAddr, jint prec,
                                                                                                                    jint method)
{
    return jniDistributed<step2Master, low_order_moments::Method, Distributed, defaultDense, singlePassDense, sumDense, fastCSR, singlePassCSR,
                          sumCSR>::getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_DistributedStep2Master
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_DistributedStep2Master_cGetInput(JNIEnv * env, jobject thisObj,
                                                                                                            jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, low_order_moments::Method, Distributed, defaultDense, singlePassDense, sumDense, fastCSR, singlePassCSR,
                          sumCSR>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_DistributedStep2Master
 * Method:    cClone
 * Signature: (JII)V
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_DistributedStep2Master_cClone(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                         jint prec, jint method)
{
    return jniDistributed<step2Master, low_order_moments::Method, Distributed, defaultDense, singlePassDense, sumDense, fastCSR, singlePassCSR,
                          sumCSR>::getClone(prec, method, algAddr);
}
