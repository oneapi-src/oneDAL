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
#include "kmeans/JParameter.h"

using namespace daal::algorithms::kmeans;

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    initEuclidean
 * Signature:(JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_initEuclidean
(JNIEnv *, jobject, jlong nClusters, jlong maxIterations)
{
    return(jlong)(new Parameter(nClusters, maxIterations));
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cGetNClusters
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cGetNClusters
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((Parameter *)parameterAddress)->nClusters;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cGetMaxIterations
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cGetMaxIterations
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((Parameter *)parameterAddress)->maxIterations;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cGetAccuracyThreshold
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cGetAccuracyThreshold
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((Parameter *)parameterAddress)->accuracyThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cGetGamma
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cGetGamma
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((Parameter *)parameterAddress)->gamma;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cGetAssignFlag
 * Signature:(J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cGetAssignFlag
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((Parameter *)parameterAddress)->assignFlag;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cSetNClusters
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cSetNClusters
(JNIEnv *, jobject, jlong parameterAddress, jlong nClusters)
{
    ((Parameter *)parameterAddress)->nClusters = nClusters;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cSetMaxIterations
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cSetMaxIterations
(JNIEnv *, jobject, jlong parameterAddress, jlong maxIterations)
{
    ((Parameter *)parameterAddress)->maxIterations = maxIterations;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cSetAccuracyThreshold
 * Signature:(JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cSetAccuracyThreshold
(JNIEnv *, jobject, jlong parameterAddress, jdouble accuracyThreshold)
{
    ((Parameter *)parameterAddress)->accuracyThreshold = accuracyThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cSetGamma
 * Signature:(JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cSetGamma
(JNIEnv *, jobject, jlong parameterAddress, jdouble gamma)
{
    ((Parameter *)parameterAddress)->gamma = gamma;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cSetAssignFlag
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cSetAssignFlag
(JNIEnv *, jobject, jlong parameterAddress, jboolean assignFlag)
{
    ((Parameter *)parameterAddress)->assignFlag = assignFlag;
}
