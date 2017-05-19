/* file: truncated_gaussian_batch.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
#include "neural_networks/initializers/truncated_gaussian/JTruncatedGaussianBatch.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_truncated_gaussian_TruncatedGaussianBatch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_truncated_1gaussian_TruncatedGaussianBatch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method, jdouble mean, jdouble sigma, jlong seed)
{
    return jniBatch<initializers::truncated_gaussian::Method, initializers::truncated_gaussian::Batch, initializers::truncated_gaussian::defaultDense>::newObj(
               prec, method, mean, sigma, seed);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_truncated_gaussian_TruncatedGaussianBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_truncated_1gaussian_TruncatedGaussianBatch_cInitParameter
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<initializers::truncated_gaussian::Method, initializers::truncated_gaussian::Batch, initializers::truncated_gaussian::defaultDense>::getParameter(
        prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_truncated_gaussian_TruncatedGaussianBatch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_truncated_1gaussian_TruncatedGaussianBatch_cGetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<initializers::truncated_gaussian::Method, initializers::truncated_gaussian::Batch, initializers::truncated_gaussian::defaultDense>::getResult(
        prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_truncated_gaussian_TruncatedGaussianBatch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_truncated_1gaussian_TruncatedGaussianBatch_cClone
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<initializers::truncated_gaussian::Method, initializers::truncated_gaussian::Batch, initializers::truncated_gaussian::defaultDense>::getClone(
        prec, method, algAddr);
}
