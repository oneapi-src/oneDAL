/* file: batch.cpp */
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
#include "em_gmm/JBatch.h"
#include "em_gmm/JMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::em_gmm;

#define DefaultDense    com_intel_daal_algorithms_em_gmm_Method_defaultDenseValue

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Batch
 * Method:    cInit
 * Signature:(IIJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_Batch_cInit
(JNIEnv *, jobject, jint precision, jint method, jlong nComponents)
{
    return jniBatch<em_gmm::Method, Batch, defaultDense>::newObj(precision, method, nComponents);
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Batch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_Batch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<em_gmm::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Batch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_Batch_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniBatch<em_gmm::Method, Batch, defaultDense>::setResult<em_gmm::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Batch
 * Method:    cInitParameter
 * Signature: (JII)V
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_Batch_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<em_gmm::Method, Batch, defaultDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Batch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_Batch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<em_gmm::Method, Batch, defaultDense>::getClone(prec, method, algAddr);
}
