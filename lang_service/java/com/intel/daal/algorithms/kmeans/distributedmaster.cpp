/* file: distributedmaster.cpp */
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

#include "daal.h"
#include "kmeans/JDistributedStep2Master.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::kmeans;


JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_DistributedStep2Master_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method, jlong nClusters)
{
    return jniDistributed<step2Master,kmeans::Method,Distributed,lloydDense,lloydCSR>::newObj(prec,method,nClusters);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_DistributedStep2Master_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master,kmeans::Method,Distributed,lloydDense,lloydCSR>::getInput(prec,method,algAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_DistributedStep2Master_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master,kmeans::Method,Distributed,lloydDense,lloydCSR>::getResult(prec,method,algAddr);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_DistributedStep2Master_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniDistributed<step2Master,kmeans::Method,Distributed,lloydDense,lloydCSR>::setResult<kmeans::Result>(prec,method,algAddr,resultAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_DistributedStep2Master_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master,kmeans::Method,Distributed,lloydDense,lloydCSR>::getPartialResult(prec,method,algAddr);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_DistributedStep2Master_cSetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong partialResultAddr, jboolean initFlag)
{
    jniDistributed<step2Master,kmeans::Method,Distributed,lloydDense,lloydCSR>::
        setPartialResult<kmeans::PartialResult>(prec,method,algAddr,partialResultAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_DistributedStep2Master_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master,kmeans::Method,Distributed,lloydDense,lloydCSR>::getClone(prec,method,algAddr);
}
