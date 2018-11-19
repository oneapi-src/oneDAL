/* file: batch.cpp */
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
#include "low_order_moments/JBatch.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::low_order_moments;

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_Offline
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_Batch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<low_order_moments::Method, Batch, defaultDense, singlePassDense, sumDense, fastCSR, singlePassCSR, sumCSR>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_Batch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_Batch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<low_order_moments::Method, Batch, defaultDense, singlePassDense, sumDense, fastCSR, singlePassCSR, sumCSR>::
        getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_Batch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_Batch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<low_order_moments::Method, Batch, defaultDense, singlePassDense, sumDense, fastCSR, singlePassCSR, sumCSR>::
        getClone(prec, method, algAddr);
}
