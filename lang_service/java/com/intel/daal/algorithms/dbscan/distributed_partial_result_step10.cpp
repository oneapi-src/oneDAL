/* file: distributed_partial_result_step10.cpp */
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
#include "com_intel_daal_algorithms_dbscan_DistributedPartialResultStep10.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::dbscan;

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedPartialResultStep10_cNewDistributedPartialResultStep10
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<DistributedPartialResultStep10>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_DistributedPartialResultStep10
 * Method:    cGetNumericTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedPartialResultStep10_cGetNumericTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<DistributedPartialResultStep10>::get<DistributedPartialResultStep10NumericTableId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_DistributedPartialResultStep10
 * Method:    cSetNumericTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedPartialResultStep10_cSetNumericTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<DistributedPartialResultStep10>::set<DistributedPartialResultStep10NumericTableId, NumericTable>(resAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_DistributedPartialResultStep10
 * Method:    cGetDataCollection
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedPartialResultStep10_cGetDataCollection
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<DistributedPartialResultStep10>::get<DistributedPartialResultStep10CollectionId, DataCollection>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_DistributedPartialResultStep10
 * Method:    cSetDataCollection
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedPartialResultStep10_cSetDataCollection
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong dcAddr)
{
    jniArgument<DistributedPartialResultStep10>::set<DistributedPartialResultStep10CollectionId, DataCollection>(resAddr, id, dcAddr);
}
