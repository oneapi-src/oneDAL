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
#include "normalization/zscore/JBatch.h"
#include "normalization/zscore/JMethod.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::normalization::zscore;

/*
 * Class:     com_intel_daal_algorithms_normalization_zscore_Batch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_normalization_zscore_Batch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<normalization::zscore::Method, Batch, defaultDense, sumDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_normalization_zscore_Batch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_normalization_zscore_Batch_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<normalization::zscore::Method, Batch, defaultDense, sumDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_normalization_zscore_Batch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_normalization_zscore_Batch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<normalization::zscore::Method, Batch, defaultDense, sumDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_normalization_zscore_Batch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_normalization_zscore_Batch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<normalization::zscore::Method, Batch, defaultDense, sumDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_normalization_zscore_Batch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_normalization_zscore_Batch_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniBatch<normalization::zscore::Method, Batch, defaultDense, sumDense>::
        setResult<normalization::zscore::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_normalization_zscore_Batch
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_normalization_zscore_Batch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<normalization::zscore::Method, Batch, defaultDense, sumDense>::getClone(prec, method, algAddr);
}
