/* file: parameter.cpp */
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
#include "em_gmm/JMethod.h"
#include "em_gmm/JParameter.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::em_gmm;

#define DefaultDense com_intel_daal_algorithms_em_gmm_Method_defaultDenseValue

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

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cGetRegularizationFactor
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cGetRegularizationFactor
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((em_gmm::Parameter *)parameterAddress)->regularizationFactor;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cSetRegularizationFactor
 * Signature:(JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cSetRegularizationFactor
(JNIEnv *, jobject, jlong parameterAddress, jdouble regularizationFactor)
{
    ((em_gmm::Parameter *)parameterAddress)->regularizationFactor = regularizationFactor;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cGetCovarianceStorage
 * Signature:(J)D
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cGetCovarianceStorage
(JNIEnv *, jobject, jlong parameterAddress)
{
    return (jint)((em_gmm::Parameter *)parameterAddress)->covarianceStorage;
}

/*
 * Class:     com_intel_daal_algorithms_em_gmm_Parameter
 * Method:    cGetCovarianceStorage
 * Signature: (J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_em_1gmm_Parameter_cSetCovarianceStorage
(JNIEnv *, jobject, jlong parameterAddress, jint covarianceStorage)
{
    ((em_gmm::Parameter *)parameterAddress)->covarianceStorage = (em_gmm::CovarianceStorageId)covarianceStorage;
}
