/* file: distributed_step7_master_input.cpp */
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
#include "com_intel_daal_algorithms_dbscan_DistributedStep7MasterInput.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::dbscan;

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep7MasterInput
* Method:    cSetDataCollection
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep7MasterInput_cSetDataCollection(JNIEnv *, jobject, jlong inputAddr,
                                                                                                            jint id, jlong dcAddr)
{
    jniInput<DistributedInput<step7Master> >::set<Step7MasterCollectionInputId, DataCollection>(inputAddr, id, dcAddr);
}

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep7MasterInput
* Method:    cAddNumericTable
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep7MasterInput_cAddNumericTable(JNIEnv *, jobject, jlong inputAddr, jint id,
                                                                                                          jlong ntAddr)
{
    jniInput<DistributedInput<step7Master> >::add<Step7MasterCollectionInputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep7MasterInput
* Method:    cGetDataCollection
* Signature:(JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep7MasterInput_cGetDataCollection(JNIEnv *, jobject, jlong inputAddr,
                                                                                                             jint id)
{
    return jniInput<DistributedInput<step7Master> >::get<Step7MasterCollectionInputId, DataCollection>(inputAddr, id);
}
