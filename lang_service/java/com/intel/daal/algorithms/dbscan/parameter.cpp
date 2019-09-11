/* file: parameter.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
#include "com_intel_daal_algorithms_dbscan_Parameter.h"

using namespace daal::algorithms::dbscan;

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cGetEpsilon
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cGetEpsilon
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((Parameter *)parameterAddress)->epsilon;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cGetMinObservations
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cGetMinObservations
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((Parameter *)parameterAddress)->minObservations;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cGetMemorySavingMode
 * Signature:(J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cGetMemorySavingMode
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((Parameter *)parameterAddress)->memorySavingMode;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cGetResultsToCompute
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cGetResultsToCompute
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((Parameter *)parameterAddress)->resultsToCompute;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cGetBlockIndex
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cGetBlockIndex
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((Parameter *)parameterAddress)->blockIndex;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cGetNBlocks
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cGetNBlocks
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((Parameter *)parameterAddress)->nBlocks;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cGetLeftBlocks
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cGetLeftBlocks
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((Parameter *)parameterAddress)->leftBlocks;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cGetRightBlocks
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cGetRightBlocks
(JNIEnv *, jobject, jlong parameterAddress)
{
    return((Parameter *)parameterAddress)->rightBlocks;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cSetEpsilon
 * Signature:(JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cSetEpsilon
(JNIEnv *, jobject, jlong parameterAddress, jdouble epsilon)
{
    ((Parameter *)parameterAddress)->epsilon = epsilon;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cSetMinObservations
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cSetMinObservations
(JNIEnv *, jobject, jlong parameterAddress, jlong minObservations)
{
    ((Parameter *)parameterAddress)->minObservations = minObservations;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cSetMemorySavingMode
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cSetMemorySavingMode
(JNIEnv *, jobject, jlong parameterAddress, jboolean memorySavingMode)
{
    ((Parameter *)parameterAddress)->memorySavingMode = memorySavingMode;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cSetResultsToCompute
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cSetResultsToCompute
(JNIEnv *, jobject, jlong parameterAddress, jlong resultsToCompute)
{
    ((Parameter *)parameterAddress)->resultsToCompute = resultsToCompute;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cSetBlockIndex
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cSetBlockIndex
(JNIEnv *, jobject, jlong parameterAddress, jlong blockIndex)
{
    ((Parameter *)parameterAddress)->blockIndex = blockIndex;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cSetNBlocks
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cSetNBlocks
(JNIEnv *, jobject, jlong parameterAddress, jlong nBlocks)
{
    ((Parameter *)parameterAddress)->nBlocks = nBlocks;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cSetLeftBlocks
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cSetLeftBlocks
(JNIEnv *, jobject, jlong parameterAddress, jlong leftBlocks)
{
    ((Parameter *)parameterAddress)->leftBlocks = leftBlocks;
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_Parameter
 * Method:    cSetRightBlocks
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_Parameter_cSetRightBlocks
(JNIEnv *, jobject, jlong parameterAddress, jlong rightBlocks)
{
    ((Parameter *)parameterAddress)->rightBlocks = rightBlocks;
}
