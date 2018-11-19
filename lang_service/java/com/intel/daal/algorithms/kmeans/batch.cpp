/* file: batch.cpp */
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
