/* file: distributed_partial_result_step9.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
#include "com_intel_daal_algorithms_dbscan_DistributedPartialResultStep9.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::dbscan;

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedPartialResultStep9_cNewDistributedPartialResultStep9(JNIEnv * env,
                                                                                                                              jobject thisObj)
{
    return jniArgument<DistributedPartialResultStep9>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_DistributedPartialResultStep9
 * Method:    cGetDataCollection
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedPartialResultStep9_cGetDataCollection(JNIEnv * env, jobject thisObj,
                                                                                                               jlong resAddr, jint id)
{
    return jniArgument<DistributedPartialResultStep9>::get<DistributedPartialResultStep9Id, DataCollection>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_dbscan_DistributedPartialResultStep9
 * Method:    cSetDataCollection
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedPartialResultStep9_cSetDataCollection(JNIEnv * env, jobject thisObj,
                                                                                                              jlong resAddr, jint id, jlong dcAddr)
{
    jniArgument<DistributedPartialResultStep9>::set<DistributedPartialResultStep9Id, DataCollection>(resAddr, id, dcAddr);
}
