/* file: distributedmaster.cpp */
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
#include "low_order_moments/JDistributedStep2Master.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::low_order_moments;

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_Distributed
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_DistributedStep2Master_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniDistributed<step2Master, low_order_moments::Method, Distributed, defaultDense, singlePassDense, sumDense,
        fastCSR, singlePassCSR, sumCSR>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_DistributedStep2Master
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_DistributedStep2Master_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniDistributed<step2Master, low_order_moments::Method, Distributed, defaultDense, singlePassDense, sumDense,
        fastCSR, singlePassCSR, sumCSR>::setResult<low_order_moments::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_DistributedStep2Master
 * Method:    cSetPartialResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_DistributedStep2Master_cSetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong partialResultAddr, jboolean initFlag)
{
    jniDistributed<step2Master, low_order_moments::Method, Distributed, defaultDense, singlePassDense, sumDense,
        fastCSR, singlePassCSR, sumCSR>::setPartialResult<low_order_moments::PartialResult>(prec, method, algAddr, partialResultAddr, initFlag);
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_DistributedStep2Master
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_DistributedStep2Master_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, low_order_moments::Method, Distributed, defaultDense, singlePassDense, sumDense,
        fastCSR, singlePassCSR, sumCSR>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_DistributedStep2Master
 * Method:    cGetPartialResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_DistributedStep2Master_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, low_order_moments::Method, Distributed, defaultDense, singlePassDense, sumDense,
        fastCSR, singlePassCSR, sumCSR>::getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_DistributedStep2Master
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_DistributedStep2Master_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, low_order_moments::Method, Distributed, defaultDense, singlePassDense, sumDense,
        fastCSR, singlePassCSR, sumCSR>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_DistributedStep2Master
 * Method:    cClone
 * Signature: (JII)V
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_DistributedStep2Master_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, low_order_moments::Method, Distributed, defaultDense, singlePassDense, sumDense,
        fastCSR, singlePassCSR, sumCSR>::getClone(prec, method, algAddr);
}
