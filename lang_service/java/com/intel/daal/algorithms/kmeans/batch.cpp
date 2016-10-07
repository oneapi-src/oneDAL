/* file: batch.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
#include "kmeans/JBatch.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::kmeans;

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_Batch_cInit
(JNIEnv *, jobject, jint prec, jint method, jlong nClusters, jlong maxIterations)
{
    return jniBatch<kmeans::Method,Batch,lloydDense,lloydCSR>::newObj(prec,method,nClusters,maxIterations);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_Batch_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<kmeans::Method,Batch,lloydDense,lloydCSR>::getParameter(prec,method,algAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_Batch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<kmeans::Method,Batch,lloydDense,lloydCSR>::getInput(prec,method,algAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_Batch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<kmeans::Method,Batch,lloydDense,lloydCSR>::getResult(prec,method,algAddr);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_Batch_cSetResult
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniBatch<kmeans::Method,Batch,lloydDense,lloydCSR>::setResult<kmeans::Result>(prec,method,algAddr,resultAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_Batch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<kmeans::Method,Batch,lloydDense,lloydCSR>::getClone(prec,method,algAddr);
}
