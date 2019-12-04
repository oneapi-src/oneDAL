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
#include "com_intel_daal_algorithms_dbscan_Parameter.h"

using namespace daal::algorithms::dbscan;

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cGetEpsilon
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cGetEpsilon(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((Parameter *)parameterAddress)->epsilon;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cGetMinObservations
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cGetMinObservations(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((Parameter *)parameterAddress)->minObservations;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cGetMemorySavingMode
 * Signature:(J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cGetMemorySavingMode(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((Parameter *)parameterAddress)->memorySavingMode;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cGetResultsToCompute
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cGetResultsToCompute(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((Parameter *)parameterAddress)->resultsToCompute;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cGetBlockIndex
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cGetBlockIndex(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((Parameter *)parameterAddress)->blockIndex;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cGetNBlocks
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cGetNBlocks(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((Parameter *)parameterAddress)->nBlocks;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cGetLeftBlocks
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cGetLeftBlocks(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((Parameter *)parameterAddress)->leftBlocks;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cGetRightBlocks
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cGetRightBlocks(JNIEnv *, jobject, jlong parameterAddress)
{
    return ((Parameter *)parameterAddress)->rightBlocks;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cSetEpsilon
 * Signature:(JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cSetEpsilon(JNIEnv *, jobject, jlong parameterAddress, jdouble epsilon)
{
    ((Parameter *)parameterAddress)->epsilon = epsilon;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cSetMinObservations
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cSetMinObservations(JNIEnv *, jobject, jlong parameterAddress,
                                                                                           jlong minObservations)
{
    ((Parameter *)parameterAddress)->minObservations = minObservations;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cSetMemorySavingMode
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cSetMemorySavingMode(JNIEnv *, jobject, jlong parameterAddress,
                                                                                            jboolean memorySavingMode)
{
    ((Parameter *)parameterAddress)->memorySavingMode = memorySavingMode;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cSetResultsToCompute
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cSetResultsToCompute(JNIEnv *, jobject, jlong parameterAddress,
                                                                                            jlong resultsToCompute)
{
    ((Parameter *)parameterAddress)->resultsToCompute = resultsToCompute;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cSetBlockIndex
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cSetBlockIndex(JNIEnv *, jobject, jlong parameterAddress, jlong blockIndex)
{
    ((Parameter *)parameterAddress)->blockIndex = blockIndex;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cSetNBlocks
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cSetNBlocks(JNIEnv *, jobject, jlong parameterAddress, jlong nBlocks)
{
    ((Parameter *)parameterAddress)->nBlocks = nBlocks;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cSetLeftBlocks
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cSetLeftBlocks(JNIEnv *, jobject, jlong parameterAddress, jlong leftBlocks)
{
    ((Parameter *)parameterAddress)->leftBlocks = leftBlocks;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cSetRightBlocks
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cSetRightBlocks(JNIEnv *, jobject, jlong parameterAddress, jlong rightBlocks)
{
    ((Parameter *)parameterAddress)->rightBlocks = rightBlocks;
}
