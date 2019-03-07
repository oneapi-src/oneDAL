/* file: distributed.cpp */
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

#include "svd/JDistributedStep2Master.h"
#include "svd/JDistributedStep3Local.h"
#include "svd/JMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::svd;

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep2Master
 * Method:    cInitDistributed
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_DistributedStep2Master_cInitDistributed
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniDistributed<step2Master, svd::Method, Distributed, defaultDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep2Master
 * Method:    cInitParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_DistributedStep2Master_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, svd::Method, Distributed, defaultDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep2Master
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_DistributedStep2Master_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, svd::Method, Distributed, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep2Master
 * Method:    cInitParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_DistributedStep2Master_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, svd::Method, Distributed, defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep2Master
 * Method:    cGetPartialResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_DistributedStep2Master_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, svd::Method, Distributed, defaultDense>::getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep2Master
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_DistributedStep2Master_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, svd::Method, Distributed, defaultDense>::getClone(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep3Local
 * Method:    cInitDistributed
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_DistributedStep3Local_cInitDistributed
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniDistributed<step3Local, svd::Method, Distributed, defaultDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep3Local
 * Method:    cInitParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_DistributedStep3Local_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step3Local, svd::Method, Distributed, defaultDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep3Local
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_DistributedStep3Local_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step3Local, svd::Method, Distributed, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep3Local
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_DistributedStep3Local_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step3Local, svd::Method, Distributed, defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep3Local
 * Method:    cGetPartialResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_DistributedStep3Local_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step3Local, svd::Method, Distributed, defaultDense>::getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_svd_DistributedStep3Local
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svd_DistributedStep3Local_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step3Local, svd::Method, Distributed, defaultDense>::getClone(prec, method, algAddr);
}
