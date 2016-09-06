/* file: initparameter.cpp */
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
#include "kmeans/init/JInitParameter.h"

using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    initEuclidean
 * Signature:(JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitParameter_init
(JNIEnv *, jobject, jlong nClusters, jlong startingIndex)
{
    return(jlong)(new kmeans::init::Parameter(nClusters, startingIndex));
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cGetNClusters
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitParameter_cGetNClusters
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((kmeans::init::Parameter *)parameterAddress)->nClusters;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitParameter_cGetNRowsTotal
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((kmeans::init::Parameter *)parameterAddress)->nRowsTotal;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cGetMaxIterations
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitParameter_cGetOffset
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((kmeans::init::Parameter *)parameterAddress)->offset;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cSetNClusters
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitParameter_cSetNRowsTotal
(JNIEnv *, jobject, jlong parameterAddress, jlong nRowsTotal)
{
    ((kmeans::init::Parameter *)parameterAddress)->nRowsTotal = nRowsTotal;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitParameter_cSetNClusters
(JNIEnv *, jobject, jlong parameterAddress, jlong nClusters)
{
    ((kmeans::init::Parameter *)parameterAddress)->nClusters = nClusters;
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_Parameter
 * Method:    cSetMaxIterations
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitParameter_cSetOffset
(JNIEnv *, jobject, jlong parameterAddress, jlong offset)
{
    ((kmeans::init::Parameter *)parameterAddress)->offset = offset;
}
