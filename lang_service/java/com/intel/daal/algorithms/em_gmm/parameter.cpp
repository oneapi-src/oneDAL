/* file: parameter.cpp */
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
#include "em_gmm/JMethod.h"
#include "em_gmm/JParameter.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::em_gmm;

#define DefaultDense com_intel_daal_algorithms_em_gmm_Method_defaultDenseValue

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cInit
 * Signature: (JIIIJJD)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cInit
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jint cmode, jlong nComponents, jlong maxIterations, jdouble accuracyThreshold)
{
    em_gmm::Parameter *parameterAddr = (em_gmm::Parameter *)jniBatch<em_gmm::Method, Batch, defaultDense>::getParameter(prec, method, algAddr);

    if(parameterAddr)
    {
        parameterAddr->nComponents = nComponents;
        parameterAddr->maxIterations = maxIterations;
        parameterAddr->accuracyThreshold = accuracyThreshold;
    }

    return (jlong)parameterAddr;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cGetNComponents
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cGetNComponents
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((em_gmm::Parameter *)parameterAddress)->nComponents;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cGetMaxIterations
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cGetMaxIterations
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((em_gmm::Parameter *)parameterAddress)->maxIterations;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cGetAccuracyThreshold
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cGetAccuracyThreshold
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((em_gmm::Parameter *)parameterAddress)->accuracyThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cSetNComponents
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cSetNComponents
(JNIEnv *, jobject, jlong parameterAddress, jlong nComponents)
{
    ((em_gmm::Parameter *)parameterAddress)->nComponents = nComponents;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cSetMaxIterations
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cSetMaxIterations
(JNIEnv *, jobject, jlong parameterAddress, jlong maxIterations)
{
    ((em_gmm::Parameter *)parameterAddress)->maxIterations = maxIterations;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cSetAccuracyThreshold
 * Signature:(JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cSetAccuracyThreshold
(JNIEnv *, jobject, jlong parameterAddress, jdouble accuracyThreshold)
{
    ((em_gmm::Parameter *)parameterAddress)->accuracyThreshold = accuracyThreshold;
}
