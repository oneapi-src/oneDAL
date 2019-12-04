/* file: initdistributedlocal.cpp */
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
#include "com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local.h"
#include "com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Local.h"
#include "com_intel_daal_algorithms_kmeans_init_InitDistributedStep4Local.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::kmeans::init;

#define AllMethodsList                                                                                                                              \
    kmeans::init::Method, Distributed, deterministicDense, randomDense, plusPlusDense, parallelPlusDense, deterministicCSR, randomCSR, plusPlusCSR, \
        parallelPlusCSR

#define PlusPlusMethodsList kmeans::init::Method, Distributed, plusPlusDense, parallelPlusDense, plusPlusCSR, parallelPlusCSR
/*
 * Class:     com_intel_daal_algorithms_kmeans_Distributed
 * Method:    cInit
 * Signature:(IIJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local_cInit(JNIEnv * env, jobject thisObj, jint prec,
                                                                                                   jint method, jlong nClusters, jlong offset)
{
    return jniDistributed<step1Local, AllMethodsList>::newObj(prec, method, nClusters, offset);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep1Local
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local_cInitParameter(JNIEnv * env, jobject thisObj,
                                                                                                            jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local, AllMethodsList>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep1Local
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local_cGetInput(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                       jint prec, jint method)
{
    return jniDistributed<step1Local, AllMethodsList>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep1Local
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local_cSetResult(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                       jint prec, jint method, jlong resultAddr)
{
    jniDistributed<step1Local, AllMethodsList>::setResult<kmeans::init::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep1Local
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local_cGetResult(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                        jint prec, jint method)
{
    return jniDistributed<step1Local, AllMethodsList>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep1Local
 * Method:    cSetPartialResult
 * Signature: (JIIJZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local_cSetPartialResult(JNIEnv * env, jobject thisObj,
                                                                                                              jlong algAddr, jint prec, jint method,
                                                                                                              jlong partialResultAddr,
                                                                                                              jboolean initFlag)
{
    jniDistributed<step1Local, AllMethodsList>::setPartialResult<kmeans::init::PartialResult>(prec, method, algAddr, partialResultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep1Local
 * Method:    cGetPartialResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local_cGetPartialResult(JNIEnv * env, jobject thisObj,
                                                                                                               jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local, AllMethodsList>::getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep1Local
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep1Local_cClone(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                    jint prec, jint method)
{
    return jniDistributed<step1Local, AllMethodsList>::getClone(prec, method, algAddr);
}

/////////////////////////////////////// plusPlus methods ///////////////////////////////////////////////////////
///////////////////////////////////////   step2Local     ///////////////////////////////////////////////////////
/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Local
* Method:    cInit
* Signature: (IIJZ)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Local_cInit(JNIEnv * env, jobject thisObj, jint prec,
                                                                                                   jint method, jlong nClusters,
                                                                                                   jboolean bFirstIteration)
{
    return jniDistributed<step2Local, PlusPlusMethodsList>::newObj(prec, method, nClusters, bFirstIteration);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Local
* Method:    cInitParameter
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Local_cInitParameter(JNIEnv * env, jobject thisObj,
                                                                                                            jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Local, PlusPlusMethodsList>::getParameter(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Local
* Method:    cGetInput
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Local_cGetInput(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                       jint prec, jint method)
{
    return jniDistributed<step2Local, PlusPlusMethodsList>::getInput(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Local
* Method:    cSetPartialResult
* Signature: (JIIJZ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Local_cSetPartialResult(JNIEnv * env, jobject thisObj,
                                                                                                              jlong algAddr, jint prec, jint method,
                                                                                                              jlong partialResultAddr,
                                                                                                              jboolean initFlag)
{
    jniDistributed<step2Local, PlusPlusMethodsList>::setPartialResult<kmeans::init::DistributedStep2LocalPlusPlusPartialResult>(prec, method, algAddr,
                                                                                                                                partialResultAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Local
* Method:    cGetPartialResult
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Local_cGetPartialResult(JNIEnv * env, jobject thisObj,
                                                                                                               jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Local, PlusPlusMethodsList>::getPartialResult(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Local
* Method:    cClone
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Local_cClone(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                    jint prec, jint method)
{
    return jniDistributed<step2Local, PlusPlusMethodsList>::getClone(prec, method, algAddr);
}

///////////////////////////////////////   step4Local     ///////////////////////////////////////////////////////
/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep4Local
* Method:    cInit
* Signature: (IIJ)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep4Local_cInit(JNIEnv * env, jobject thisObj, jint prec,
                                                                                                   jint method, jlong nClusters)
{
    return jniDistributed<step4Local, PlusPlusMethodsList>::newObj(prec, method, nClusters);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep4Local
* Method:    cInitParameter
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep4Local_cInitParameter(JNIEnv * env, jobject thisObj,
                                                                                                            jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step4Local, PlusPlusMethodsList>::getParameter(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep4Local
* Method:    cGetInput
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep4Local_cGetInput(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                       jint prec, jint method)
{
    return jniDistributed<step4Local, PlusPlusMethodsList>::getInput(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep4Local
* Method:    cSetPartialResult
* Signature: (JIIJZ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep4Local_cSetPartialResult(JNIEnv * env, jobject thisObj,
                                                                                                              jlong algAddr, jint prec, jint method,
                                                                                                              jlong partialResultAddr,
                                                                                                              jboolean initFlag)
{
    jniDistributed<step4Local, PlusPlusMethodsList>::setPartialResult<kmeans::init::DistributedStep4LocalPlusPlusPartialResult>(prec, method, algAddr,
                                                                                                                                partialResultAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep4Local
* Method:    cGetPartialResult
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep4Local_cGetPartialResult(JNIEnv * env, jobject thisObj,
                                                                                                               jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step4Local, PlusPlusMethodsList>::getPartialResult(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep4Local
* Method:    cClone
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep4Local_cClone(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                    jint prec, jint method)
{
    return jniDistributed<step4Local, PlusPlusMethodsList>::getClone(prec, method, algAddr);
}
