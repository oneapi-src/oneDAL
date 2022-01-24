/* file: bacon_batch.cpp */
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_multivariate_outlier_detection_bacondense_Batch */

#include "daal.h"
#include "com_intel_daal_algorithms_multivariate_outlier_detection_bacondense_Batch.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::bacon_outlier_detection;

/*
 * Class:     com_intel_daal_algorithms_multivariate_outlier_detection_bacondense_Batch \DAAL_DEPRECATED
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_bacondense_Batch_cInit(JNIEnv * env, jobject thisObj,
                                                                                                               jint prec, jint method)
{
    return jniBatch<bacon_outlier_detection::Method, Batch, defaultDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_multivariate_outlier_detection_bacondense_Batch \DAAL_DEPRECATED
 * Method:    cInitParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_bacondense_Batch_cInitParameter(JNIEnv * env, jobject thisObj,
                                                                                                                        jlong algAddr, jint prec,
                                                                                                                        jint method)
{
    return jniBatch<bacon_outlier_detection::Method, Batch, defaultDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multivariate_outlier_detection_bacondense_Batch \DAAL_DEPRECATED
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_bacondense_Batch_cGetInput(JNIEnv * env, jobject thisObj,
                                                                                                                   jlong algAddr, jint prec,
                                                                                                                   jint method)
{
    return jniBatch<bacon_outlier_detection::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multivariate_outlier_detection_bacondense_Batch \DAAL_DEPRECATED
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_bacondense_Batch_cGetResult(JNIEnv * env, jobject thisObj,
                                                                                                                    jlong algAddr, jint prec,
                                                                                                                    jint method)
{
    return jniBatch<bacon_outlier_detection::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multivariate_outlier_detection_bacondense_Batch \DAAL_DEPRECATED
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_bacondense_Batch_cSetResult(JNIEnv * env, jobject thisObj,
                                                                                                                   jlong algAddr, jint prec,
                                                                                                                   jint method, jlong resultAddr)
{
    jniBatch<bacon_outlier_detection::Method, Batch, defaultDense>::setResult<bacon_outlier_detection::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multivariate_outlier_detection_bacondense_Batch \DAAL_DEPRECATED
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multivariate_1outlier_1detection_bacondense_Batch_cClone(JNIEnv * env, jobject thisObj,
                                                                                                                jlong algAddr, jint prec, jint method)
{
    return jniBatch<bacon_outlier_detection::Method, Batch, defaultDense>::getClone(prec, method, algAddr);
}
