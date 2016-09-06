/* file: distributedlocal.cpp */
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
#include "kmeans/JDistributedStep1Local.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::kmeans;
using namespace daal::data_management;


JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_DistributedStep1Local_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method, jlong nClusters)
{
    return jniDistributed<step1Local,kmeans::Method,Distributed,lloydDense,lloydCSR>::newObj(prec,method,nClusters);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_DistributedStep1Local_cInitParameter
(JNIEnv *env, jobject thisObj, jlong addr, jint prec, jint method)
{
    return jniDistributed<step1Local,kmeans::Method,Distributed,lloydDense,lloydCSR>::getParameter(prec,method,addr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_DistributedStep1Local_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local,kmeans::Method,Distributed,lloydDense,lloydCSR>::getInput(prec,method,algAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_DistributedStep1Local_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local,kmeans::Method,Distributed,lloydDense,lloydCSR>::getResult(prec,method,algAddr);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_DistributedStep1Local_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniDistributed<step1Local,kmeans::Method,Distributed,lloydDense,lloydCSR>::setResult<kmeans::Result>(prec,method,algAddr,resultAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_DistributedStep1Local_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local,kmeans::Method,Distributed,lloydDense,lloydCSR>::getPartialResult(prec,method,algAddr);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_DistributedStep1Local_cSetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong partialResultAddr)
{
    jniDistributed<step1Local,kmeans::Method,Distributed,lloydDense,lloydCSR>::
        setPartialResult<kmeans::PartialResult>(prec,method,algAddr,partialResultAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_DistributedStep1Local_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local,kmeans::Method,Distributed,lloydDense,lloydCSR>::getClone(prec,method,algAddr);
}
