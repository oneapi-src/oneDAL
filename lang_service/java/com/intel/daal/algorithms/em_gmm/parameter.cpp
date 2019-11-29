/* file: parameter.cpp */
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
#include "com_intel_daal_algorithms_em_gmm_Parameter.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::em_gmm;

#define DefaultDense com_intel_daal_algorithms_em_gmm_Method_defaultDenseValue

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cGetNComponents
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cGetNComponents(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((em_gmm::Parameter *)parameterAddress)->nComponents;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cGetMaxIterations
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cGetMaxIterations(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((em_gmm::Parameter *)parameterAddress)->maxIterations;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cGetAccuracyThreshold
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cGetAccuracyThreshold(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((em_gmm::Parameter *)parameterAddress)->accuracyThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cSetNComponents
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cSetNComponents(JNIEnv *, jobject, jlong parameterAddress, jlong nComponents)
{
    ((em_gmm::Parameter *)parameterAddress)->nComponents = nComponents;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cSetMaxIterations
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cSetMaxIterations(JNIEnv *, jobject, jlong parameterAddress,
                                                                                          jlong maxIterations)
{
    ((em_gmm::Parameter *)parameterAddress)->maxIterations = maxIterations;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cSetAccuracyThreshold
 * Signature:(JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cSetAccuracyThreshold(JNIEnv *, jobject, jlong parameterAddress,
                                                                                              jdouble accuracyThreshold)
{
    ((em_gmm::Parameter *)parameterAddress)->accuracyThreshold = accuracyThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cGetRegularizationFactor
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cGetRegularizationFactor(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((em_gmm::Parameter *)parameterAddress)->regularizationFactor;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cSetRegularizationFactor
 * Signature:(JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cSetRegularizationFactor(JNIEnv *, jobject, jlong parameterAddress,
                                                                                                 jdouble regularizationFactor)
{
    ((em_gmm::Parameter *)parameterAddress)->regularizationFactor = regularizationFactor;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cGetCovarianceStorage
 * Signature:(J)D
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cGetCovarianceStorage(JNIEnv *, jobject, jlong parameterAddress)
{
    return (jint)((em_gmm::Parameter *)parameterAddress)->covarianceStorage;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cGetCovarianceStorage
 * Signature: (J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cSetCovarianceStorage(JNIEnv *, jobject, jlong parameterAddress,
                                                                                              jint covarianceStorage)
{
    ((em_gmm::Parameter *)parameterAddress)->covarianceStorage = (em_gmm::CovarianceStorageId)covarianceStorage;
}
