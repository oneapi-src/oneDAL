/* file: init_parameter.cpp */
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
#include "em_gmm/init/JInitParameter.h"
#include "em_gmm/init/JInitMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::em_gmm::init;

#define DefaultDense    com_intel_daal_algorithms_em_gmm_init_InitMethod_DefaultDenseValue

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitParameter
 * Method:    cInit
 * Signature: (JIIIJJJD)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitParameter_cInit
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jint cmode, jlong nComponents, jlong nTrials, jlong nDepthIter, jdouble accThr)
{
    em_gmm::init::Parameter *parameterAddr =
        (em_gmm::init::Parameter *)jniBatch<em_gmm::init::Method, Batch, defaultDense>::getParameter(prec, method, algAddr);

    if(parameterAddr)
    {
        parameterAddr->nComponents = nComponents;
        parameterAddr->nIterations = nDepthIter;
        parameterAddr->nTrials = nTrials;
        parameterAddr->accuracyThreshold = accThr;
    }

    return (jlong)parameterAddr;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitParameter
 * Method:    cGetNComponents
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitParameter_cGetNComponents
(JNIEnv *env, jobject thisObj, jlong parameterAddress)
{
    return((em_gmm::init::Parameter *)parameterAddress)->nComponents;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitParameter
 * Method:    cGetDepthIterations
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitParameter_cGetDepthIterations
(JNIEnv *env, jobject thisObj, jlong parameterAddress)
{
    return((em_gmm::init::Parameter *)parameterAddress)->nIterations;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitParameter
 * Method:    cGetNTrials
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitParameter_cGetNTrials
(JNIEnv *env, jobject thisObj, jlong parameterAddress)
{
    return((em_gmm::init::Parameter *)parameterAddress)->nTrials;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitParameter
 * Method:    cGetStartSeed
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitParameter_cGetStartSeed
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((em_gmm::init::Parameter *)parameterAddress)->seed;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitParameter
 * Method:    cGetAccuracyThreshold
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitParameter_cGetAccuracyThreshold
(JNIEnv *env, jobject thisObj, jlong parameterAddress)
{
    return((em_gmm::init::Parameter *)parameterAddress)->accuracyThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitParameter
 * Method:    cSetNComponents
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitParameter_cSetNComponents
(JNIEnv *env, jobject thisObj, jlong parameterAddress, jlong nComponents)
{
    ((em_gmm::init::Parameter *)parameterAddress)->nComponents = nComponents;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitParameter
 * Method:    cSetNDepthIterations
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitParameter_cSetNDepthIterations
(JNIEnv *env, jobject thisObj, jlong parameterAddress, jlong nIterations)
{
    ((em_gmm::init::Parameter *)parameterAddress)->nIterations = nIterations;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitParameter
 * Method:    cSetNTrials
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitParameter_cSetNTrials
(JNIEnv *env, jobject thisObj, jlong parameterAddress, jlong nTrials)
{
    ((em_gmm::init::Parameter *)parameterAddress)->nTrials = nTrials;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitParameter
 * Method:    cSetStartSeed
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitParameter_cSetStartSeed
(JNIEnv *, jobject, jlong parameterAddress, jlong seed)
{
    ((em_gmm::init::Parameter *)parameterAddress)->seed = seed;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_init_InitParameter
 * Method:    cSetAccuracyThreshold
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_init_InitParameter_cSetAccuracyThreshold
(JNIEnv *env, jobject thisObj, jlong parameterAddress, jdouble accuracyThreshold)
{
    ((em_gmm::init::Parameter *)parameterAddress)->accuracyThreshold = accuracyThreshold;
}
