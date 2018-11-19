/* file: default_batch.cpp */
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_multivariate_outlier_detection_Batch */

#include "daal.h"
#include "multivariate_outlier_detection/JBatch.h"
#include "multivariate_outlier_detection/defaultdense/JBatch.h"
#include "multivariate_outlier_detection/JMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::multivariate_outlier_detection;

/*
 * Class:     com_intel_daal_algorithms_multivariate_outlier_detection_Batch
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_Batch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<multivariate_outlier_detection::Method, Batch, defaultDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_multivariate_outlier_detection_Batch
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_Batch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<multivariate_outlier_detection::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multivariate_outlier_detection_Batch
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_Batch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<multivariate_outlier_detection::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multivariate_outlier_detection_Batch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_Batch_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, int method, jlong resultAddr)
{
    jniBatch<multivariate_outlier_detection::Method, Batch, defaultDense>::
        setResult<multivariate_outlier_detection::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multivariate_outlier_detection_Batch
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_Batch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<multivariate_outlier_detection::Method, Batch, defaultDense>::getClone(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multivariate_outlier_detection_defaultdense_Batch \DAAL_DEPRECATED
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_defaultdense_Batch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<multivariate_outlier_detection::Method, Batch, defaultDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_multivariate_outlier_detection_defaultdense_Batch \DAAL_DEPRECATED
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_defaultdense_Batch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<multivariate_outlier_detection::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multivariate_outlier_detection_defaultdense_Batch \DAAL_DEPRECATED
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_defaultdense_Batch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<multivariate_outlier_detection::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multivariate_outlier_detection_defaultdense_Batch \DAAL_DEPRECATED
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_defaultdense_Batch_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, int method, jlong resultAddr)
{
    jniBatch<multivariate_outlier_detection::Method, Batch, defaultDense>::
        setResult<multivariate_outlier_detection::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multivariate_outlier_detection_defaultdense_Batch \DAAL_DEPRECATED
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_defaultdense_Batch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<multivariate_outlier_detection::Method, Batch, defaultDense>::getClone(prec, method, algAddr);
}
