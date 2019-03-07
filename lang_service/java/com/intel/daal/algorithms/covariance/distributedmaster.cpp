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
#include "covariance/JDistributedStep2Master.h"

#include "covariance_types.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::covariance;

/*
 * Class:     com_intel_daal_algorithms_covariance_DistributedStep2Master
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_DistributedStep2Master_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniDistributed<step2Master, covariance::Method, Distributed, defaultDense, singlePassDense, sumDense,
           fastCSR, singlePassCSR, sumCSR>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_covariance_DistributedStep2Master
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_DistributedStep2Master_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, covariance::Method, Distributed, defaultDense, singlePassDense, sumDense,
           fastCSR, singlePassCSR, sumCSR>::getClone(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_covariance_DistributedStep2Master
 * Method:    cInitParameter
 * Signature:(JIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_covariance_DistributedStep2Master_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jint cmode)
{
    return jniDistributed<step2Master, covariance::Method, Distributed, defaultDense, singlePassDense, sumDense,
           fastCSR, singlePassCSR, sumCSR>::getParameter(prec, method, algAddr);
}
