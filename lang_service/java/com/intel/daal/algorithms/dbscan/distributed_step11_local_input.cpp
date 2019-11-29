/* file: distributed_step11_local_input.cpp */
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
#include "com_intel_daal_algorithms_dbscan_DistributedStep11LocalInput.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::dbscan;

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep11LocalInput
* Method:    cSetNumericTable
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep11LocalInput_cSetNumericTable(JNIEnv *, jobject, jlong inputAddr, jint id,
                                                                                                          jlong ntAddr)
{
    jniInput<DistributedInput<step11Local> >::set<Step11LocalNumericTableInputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep11LocalInput
* Method:    cGetNumericTable
* Signature:(JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep11LocalInput_cGetNumericTable(JNIEnv *, jobject, jlong inputAddr,
                                                                                                           jint id)
{
    return jniInput<DistributedInput<step11Local> >::get<Step11LocalNumericTableInputId, NumericTable>(inputAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep11LocalInput
* Method:    cSetDataCollection
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep11LocalInput_cSetDataCollection(JNIEnv *, jobject, jlong inputAddr,
                                                                                                            jint id, jlong dcAddr)
{
    jniInput<DistributedInput<step11Local> >::set<Step11LocalCollectionInputId, DataCollection>(inputAddr, id, dcAddr);
}

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep11LocalInput
* Method:    cAddNumericTable
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep11LocalInput_cAddNumericTable(JNIEnv *, jobject, jlong inputAddr, jint id,
                                                                                                          jlong ntAddr)
{
    jniInput<DistributedInput<step11Local> >::add<Step11LocalCollectionInputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_dbscan_DistributedStep11LocalInput
* Method:    cGetDataCollection
* Signature:(JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep11LocalInput_cGetDataCollection(JNIEnv *, jobject, jlong inputAddr,
                                                                                                             jint id)
{
    return jniInput<DistributedInput<step11Local> >::get<Step11LocalCollectionInputId, DataCollection>(inputAddr, id);
}
