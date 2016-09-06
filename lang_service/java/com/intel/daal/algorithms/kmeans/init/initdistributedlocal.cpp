/* file: initdistributedlocal.cpp */
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
#include "kmeans/init/JInitDistributedStep1Local.h"
#include "init_types.i"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::kmeans::init;


/*
 * Class:     com_intel_daal_algorithms_kmeans_Distributed
 * Method:    cInit
 * Signature:(IIJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method, jlong nClusters, jlong offset)
{
    return jniDistributed<step1Local,kmeans::init::Method,Distributed,deterministicDense,randomDense,deterministicCSR,randomCSR>::
        newObj(prec,method,nClusters,offset);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep1Local
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local,kmeans::init::Method,Distributed,deterministicDense,randomDense,deterministicCSR,randomCSR>::
        getParameter(prec,method,algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep1Local
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local,kmeans::init::Method,Distributed,deterministicDense,randomDense,deterministicCSR,randomCSR>::
        getInput(prec,method,algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep1Local
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniDistributed<step1Local,kmeans::init::Method,Distributed,deterministicDense,randomDense,deterministicCSR,randomCSR>::
        setResult<kmeans::init::Result>(prec,method,algAddr,resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep1Local
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local,kmeans::init::Method,Distributed,deterministicDense,randomDense,deterministicCSR,randomCSR>::
        getResult(prec,method,algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep1Local
 * Method:    cSetPartialResult
 * Signature: (JIIJZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local_cSetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong partialResultAddr, jboolean initFlag)
{
    jniDistributed<step1Local,kmeans::init::Method,Distributed,deterministicDense,randomDense,deterministicCSR,randomCSR>::
        setPartialResult<kmeans::init::PartialResult>(prec,method,algAddr,partialResultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep1Local
 * Method:    cGetPartialResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local,kmeans::init::Method,Distributed,deterministicDense,randomDense,deterministicCSR,randomCSR>::
        getPartialResult(prec,method,algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep1Local
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local,kmeans::init::Method,Distributed,deterministicDense,randomDense,deterministicCSR,randomCSR>::
        getClone(prec,method,algAddr);
}
