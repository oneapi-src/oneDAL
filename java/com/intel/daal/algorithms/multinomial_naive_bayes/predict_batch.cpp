/* file: predict_batch.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>/* Header for class com_intel_daal_algorithms_multinomial_naive_bayes_prediction_PredictionBatch */

#include "daal.h"
#include "com_intel_daal_algorithms_multinomial_naive_bayes_prediction_PredictionBatch.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::multinomial_naive_bayes::prediction;

/*
* Class:     com_intel_daal_algorithms_multinomial_naive_bayes_prediction_PredictionBatch
* Method:    cInit
* Signature: (IIJ)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_prediction_PredictionBatch_cInit(JNIEnv * env, jobject thisObj,
                                                                                                                  jint prec, jint method,
                                                                                                                  jlong nClasses)
{
    return jniBatch<multinomial_naive_bayes::prediction::Method, Batch, defaultDense, fastCSR>::newObj(prec, method, nClasses);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_prediction_PredictionBatch_cInitParameter(JNIEnv * env,
                                                                                                                           jobject thisObj,
                                                                                                                           jlong algAddr, jint prec,
                                                                                                                           jint method)
{
    return jniBatch<multinomial_naive_bayes::prediction::Method, Batch, defaultDense, fastCSR>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multinomial_naive_bayes_prediction_PredictionBatch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_prediction_PredictionBatch_cSetResult(JNIEnv * env, jobject thisObj,
                                                                                                                      jlong algAddr, jint prec,
                                                                                                                      jint method, jlong resultAddr)
{
    jniBatch<multinomial_naive_bayes::prediction::Method, Batch, defaultDense, fastCSR>::setResult<classifier::prediction::Result>(
        prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multinomial_naive_bayes_prediction_PredictionBatch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_prediction_PredictionBatch_cClone(JNIEnv * env, jobject thisObj,
                                                                                                                   jlong algAddr, jint prec,
                                                                                                                   jint method)
{
    return jniBatch<multinomial_naive_bayes::prediction::Method, Batch, defaultDense, fastCSR>::getClone(prec, method, algAddr);
}
