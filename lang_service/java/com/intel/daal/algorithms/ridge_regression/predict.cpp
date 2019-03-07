/* file: predict.cpp */
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

#include "ridge_regression/prediction/JPredictionBatch.h"
#include "ridge_regression/prediction/JPredictionMethod.h"

#include "common_helpers.h"

#define defaultDense com_intel_daal_algorithms_ridge_regression_prediction_PredictionMethod_defaultDenseValue

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::ridge_regression::prediction;

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_prediction_PredictionBatch
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_prediction_PredictionBatch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<ridge_regression::prediction::Method, Batch, defaultDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_prediction_PredictionBatch
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_prediction_PredictionBatch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<ridge_regression::prediction::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_prediction_PredictionBatch
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_prediction_PredictionBatch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<ridge_regression::prediction::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_prediction_PredictionBatch
 * Method:    cSetResult
 * Signature:(JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_ridge_1regression_prediction_PredictionBatch_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniBatch<ridge_regression::prediction::Method, Batch, defaultDense>::
        setResult<ridge_regression::prediction::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_prediction_PredictionBatch
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_prediction_PredictionBatch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<ridge_regression::prediction::Method, Batch, defaultDense>::getClone(prec, method, algAddr);
}
