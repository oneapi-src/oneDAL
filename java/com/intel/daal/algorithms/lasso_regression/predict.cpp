/* file: predict.cpp */
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

#include "com_intel_daal_algorithms_lasso_regression_prediction_PredictionBatch.h"

#include "com/intel/daal/common_helpers.h"

#include "com_intel_daal_algorithms_lasso_regression_prediction_PredictionMethod.h"
#define defaultDense com_intel_daal_algorithms_lasso_regression_prediction_PredictionMethod_defaultDenseValue

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::lasso_regression::prediction;

/*
 * Class:     com_intel_daal_algorithms_lasso_regression_prediction_PredictionBatch
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_lasso_1regression_prediction_PredictionBatch_cInit(JNIEnv * env, jobject thisObj, jint prec,
                                                                                                          jint method)
{
    return jniBatch<lasso_regression::prediction::Method, Batch, defaultDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_lasso_regression_prediction_PredictionBatch
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_lasso_1regression_prediction_PredictionBatch_cGetInput(JNIEnv * env, jobject thisObj,
                                                                                                              jlong algAddr, jint prec, jint method)
{
    return jniBatch<lasso_regression::prediction::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_lasso_regression_prediction_PredictionBatch
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_lasso_1regression_prediction_PredictionBatch_cGetResult(JNIEnv * env, jobject thisObj,
                                                                                                               jlong algAddr, jint prec, jint method)
{
    return jniBatch<lasso_regression::prediction::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_lasso_regression_prediction_PredictionBatch
 * Method:    cSetResult
 * Signature:(JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_lasso_1regression_prediction_PredictionBatch_cSetResult(JNIEnv * env, jobject thisObj,
                                                                                                              jlong algAddr, jint prec, jint method,
                                                                                                              jlong resultAddr)
{
    jniBatch<lasso_regression::prediction::Method, Batch, defaultDense>::setResult<lasso_regression::prediction::Result>(prec, method, algAddr,
                                                                                                                         resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_lasso_regression_prediction_PredictionBatch
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_lasso_1regression_prediction_PredictionBatch_cClone(JNIEnv * env, jobject thisObj,
                                                                                                           jlong algAddr, jint prec, jint method)
{
    return jniBatch<lasso_regression::prediction::Method, Batch, defaultDense>::getClone(prec, method, algAddr);
}
