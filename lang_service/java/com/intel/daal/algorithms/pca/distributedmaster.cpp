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
#include <daal.h>

#include "pca/JDistributedStep2Master.h"
#include "pca/JMethod.h"

#include "common_helpers.h"

#define CorrelationDenseValue com_intel_daal_algorithms_pca_Method_correlationDenseValue
#define SVDDenseValue         com_intel_daal_algorithms_pca_Method_svdDenseValue

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::pca;

/*
 * Class:     com_intel_daal_algorithms_pca_DistributedStep2Master
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_DistributedStep2Master_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniDistributed<step2Master, pca::Method, Distributed, correlationDense, svdDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_pca_DistributedStep2Master
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_DistributedStep2Master_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, pca::Method, Distributed, correlationDense, svdDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_DistributedStep2Master
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_DistributedStep2Master_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, pca::Method, Distributed, correlationDense, svdDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_DistributedStep2Master
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_DistributedStep2Master_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, pca::Method, Distributed, correlationDense, svdDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_DistributedStep2Master
 * Method:    cGetPartialResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_DistributedStep2Master_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, pca::Method, Distributed, correlationDense, svdDense>::getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_DistributedStep2Master
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_DistributedStep2Master_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniDistributed<step2Master, pca::Method, Distributed, correlationDense, svdDense>::setResult<pca::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_DistributedStep2Master
 * Method:    cSetPartialResult
 * Signature: (JIIJZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_DistributedStep2Master_cSetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong partialResultAddr, jboolean initFlag)
{
    jniDistributed<step2Master, pca::Method, Distributed, correlationDense, svdDense>::
        setPartialResultImpl<pca::PartialResult>(prec, method, algAddr, partialResultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_DistributedStep2Master
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_DistributedStep2Master_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, pca::Method, Distributed, correlationDense, svdDense>::getClone(prec, method, algAddr);
}
