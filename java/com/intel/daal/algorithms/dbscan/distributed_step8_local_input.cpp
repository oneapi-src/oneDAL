/* file: distributed_step8_local_input.cpp */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
#include "com_intel_daal_algorithms_dbscan_DistributedStep8LocalInput.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::dbscan;

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep8LocalInput
* Method:    cSetNumericTable
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep8LocalInput_cSetNumericTable(JNIEnv *, jobject, jlong inputAddr, jint id,
                                                                                                         jlong ntAddr)
{
    jniInput<DistributedInput<step8Local> >::set<Step8LocalNumericTableInputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep8LocalInput
* Method:    cGetNumericTable
* Signature:(JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep8LocalInput_cGetNumericTable(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<DistributedInput<step8Local> >::get<Step8LocalNumericTableInputId, NumericTable>(inputAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep8LocalInput
* Method:    cSetDataCollection
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep8LocalInput_cSetDataCollection(JNIEnv *, jobject, jlong inputAddr,
                                                                                                           jint id, jlong dcAddr)
{
    jniInput<DistributedInput<step8Local> >::set<Step8LocalCollectionInputId, DataCollection>(inputAddr, id, dcAddr);
}

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep8LocalInput
* Method:    cAddNumericTable
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep8LocalInput_cAddNumericTable(JNIEnv *, jobject, jlong inputAddr, jint id,
                                                                                                         jlong ntAddr)
{
    jniInput<DistributedInput<step8Local> >::add<Step8LocalCollectionInputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep8LocalInput
* Method:    cGetDataCollection
* Signature:(JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep8LocalInput_cGetDataCollection(JNIEnv *, jobject, jlong inputAddr,
                                                                                                            jint id)
{
    return jniInput<DistributedInput<step8Local> >::get<Step8LocalCollectionInputId, DataCollection>(inputAddr, id);
}
