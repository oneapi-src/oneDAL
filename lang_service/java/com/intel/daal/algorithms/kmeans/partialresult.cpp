/* file: partialresult.cpp */
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
#include "kmeans/JPartialResult.h"

#include "common_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::kmeans;

/*
 * Class:     com_intel_daal_algorithms_kmeans_PartialResult
 * Method:    cNewPartialResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_PartialResult_cNewPartialResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<kmeans::PartialResult>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_PartialResult
 * Method:    cGetPartialResultTable
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_PartialResult_cGetPartialResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<kmeans::PartialResult>::get<kmeans::PartialResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_PartialResult
 * Method:    cSetPartialResultTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_PartialResult_cSetPartialResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<kmeans::PartialResult>::set<kmeans::PartialResultId, NumericTable>(resAddr, id, ntAddr);
}
