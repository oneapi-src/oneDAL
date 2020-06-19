/* file: parameter.cpp */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
#include "com_intel_daal_algorithms_kmeans_Parameter.h"

using namespace daal::algorithms::kmeans;

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    initEuclidean
 * Signature:(JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_initEuclidean(JNIEnv *, jobject, jlong nClusters, jlong maxIterations)
{
    return (jlong)(new Parameter(nClusters, maxIterations));
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cGetNClusters
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cGetNClusters(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((Parameter *)parameterAddress)->nClusters;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cGetMaxIterations
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cGetMaxIterations(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((Parameter *)parameterAddress)->maxIterations;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cGetAccuracyThreshold
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cGetAccuracyThreshold(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((Parameter *)parameterAddress)->accuracyThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cGetGamma
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cGetGamma(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((Parameter *)parameterAddress)->gamma;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cGetAssignFlag
 * Signature:(J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cGetAssignFlag(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((Parameter *)parameterAddress)->assignFlag;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cGetResultsToEvaluate
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cGetResultsToEvaluate(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((Parameter *)parameterAddress)->resultsToEvaluate;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cSetNClusters
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cSetNClusters(JNIEnv *, jobject, jlong parameterAddress, jlong nClusters)
{
    ((Parameter *)parameterAddress)->nClusters = nClusters;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cSetMaxIterations
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cSetMaxIterations(JNIEnv *, jobject, jlong parameterAddress,
                                                                                         jlong maxIterations)
{
    ((Parameter *)parameterAddress)->maxIterations = maxIterations;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cSetAccuracyThreshold
 * Signature:(JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cSetAccuracyThreshold(JNIEnv *, jobject, jlong parameterAddress,
                                                                                             jdouble accuracyThreshold)
{
    ((Parameter *)parameterAddress)->accuracyThreshold = accuracyThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cSetGamma
 * Signature:(JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cSetGamma(JNIEnv *, jobject, jlong parameterAddress, jdouble gamma)
{
    ((Parameter *)parameterAddress)->gamma = gamma;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cSetAssignFlag
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cSetAssignFlag(JNIEnv *, jobject, jlong parameterAddress, jboolean assignFlag)
{
    ((Parameter *)parameterAddress)->assignFlag = assignFlag;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    SetResultsToEvaluate
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_Parameter_cSetResultsToEvaluate(JNIEnv *, jobject, jlong parameterAddress,
                                                                                             jlong resultsToEvaluate)
{
    ((Parameter *)parameterAddress)->resultsToEvaluate = resultsToEvaluate;
}
