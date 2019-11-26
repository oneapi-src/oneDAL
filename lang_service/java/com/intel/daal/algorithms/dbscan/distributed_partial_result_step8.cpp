/* file: distributed_partial_result_step8.cpp */
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
#include "com_intel_daal_algorithms_dbscan_DistributedPartialResultStep8.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::dbscan;

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedPartialResultStep8_cNewDistributedPartialResultStep8(JNIEnv * env,
                                                                                                                              jobject thisObj)
{
    return jniArgument<DistributedPartialResultStep8>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_DistributedPartialResultStep8
 * Method:    cGetNumericTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedPartialResultStep8_cGetNumericTable(JNIEnv * env, jobject thisObj,
                                                                                                             jlong resAddr, jint id)
{
    return jniArgument<DistributedPartialResultStep8>::get<DistributedPartialResultStep8NumericTableId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_DistributedPartialResultStep8
 * Method:    cSetNumericTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedPartialResultStep8_cSetNumericTable(JNIEnv * env, jobject thisObj,
                                                                                                            jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<DistributedPartialResultStep8>::set<DistributedPartialResultStep8NumericTableId, NumericTable>(resAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_DistributedPartialResultStep8
 * Method:    cGetDataCollection
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedPartialResultStep8_cGetDataCollection(JNIEnv * env, jobject thisObj,
                                                                                                               jlong resAddr, jint id)
{
    return jniArgument<DistributedPartialResultStep8>::get<DistributedPartialResultStep8CollectionId, DataCollection>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_DistributedPartialResultStep8
 * Method:    cSetDataCollection
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedPartialResultStep8_cSetDataCollection(JNIEnv * env, jobject thisObj,
                                                                                                              jlong resAddr, jint id, jlong dcAddr)
{
    jniArgument<DistributedPartialResultStep8>::set<DistributedPartialResultStep8CollectionId, DataCollection>(resAddr, id, dcAddr);
}
