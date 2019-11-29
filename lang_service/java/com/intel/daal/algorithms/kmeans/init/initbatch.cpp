/* file: initbatch.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#include "daal.h"
#include "com_intel_daal_algorithms_kmeans_init_InitBatch.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::kmeans::init;

#define MethodsList deterministicDense, randomDense, plusPlusDense, parallelPlusDense, deterministicCSR, randomCSR, plusPlusCSR, parallelPlusCSR
/*
 * Class:     com_intel_daal_algorithms_kmeans_Batch
 * Method:    cInit
 * Signature:(IIJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitBatch_cInit(JNIEnv *, jobject, jint precision, jint method, jlong nClusters)
{
    return jniBatch<kmeans::init::Method, Batch, MethodsList>::newObj(precision, method, nClusters);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Batch
 * Method:    cSetResult
 * Signature:(JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitBatch_cSetResult(JNIEnv *, jobject, jlong algAddr, jint precision, jint method,
                                                                                       jlong resultAddr)
{
    jniBatch<kmeans::init::Method, Batch, MethodsList>::setResult<kmeans::init::Result>(precision, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Batch
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitBatch_cGetResult(JNIEnv *, jobject, jlong algAddr, jint precision, jint method)
{
    return jniBatch<kmeans::init::Method, Batch, MethodsList>::getResult(precision, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Batch
 * Method:    cInitParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitBatch_cInitParameter(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<kmeans::init::Method, Batch, MethodsList>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Batch
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitBatch_cGetInput(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<kmeans::init::Method, Batch, MethodsList>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Batch
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitBatch_cClone(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<kmeans::init::Method, Batch, MethodsList>::getClone(prec, method, algAddr);
}
