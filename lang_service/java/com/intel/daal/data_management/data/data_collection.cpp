/* file: data_collection.cpp */
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

#include "JDataCollection.h"
#include "daal.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_data_DataCollection
 * Method:    cNewDataCollection
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_DataCollection_cNewDataCollection
(JNIEnv *env, jobject thisObj)
{
    data_management::DataCollection *dc = new data_management::DataCollection();
    SerializationIfacePtr *resultShPtr = new SerializationIfacePtr(dc);
    return (jlong)resultShPtr;
}

/*
 * Class:     com_intel_daal_data_1management_data_DataCollection
 * Method:    cSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_DataCollection_cSize
(JNIEnv *env, jobject thisObj, jlong dataCollectionAddr)
{
    data_management::DataCollectionPtr pDataCollection =
        services::staticPointerCast<DataCollection, SerializationIface>(
            (*(data_management::SerializationIfacePtr *)dataCollectionAddr));
    return (pDataCollection)->size();
}

/*
 * Class:     com_intel_daal_data_1management_data_DataCollection
 * Method:    cGetValue
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_DataCollection_cGetValue
(JNIEnv *env, jobject thisObj, jlong dataCollectionAddr, jlong idx)
{
    data_management::DataCollectionPtr pDataCollection =
        services::staticPointerCast<DataCollection, SerializationIface>(
            (*(data_management::SerializationIfacePtr *)dataCollectionAddr));
    data_management::SerializationIfacePtr ptr = (*pDataCollection)[idx];
    return (jlong)new data_management::SerializationIfacePtr(ptr);
}

/*
 * Class:     com_intel_daal_data_1management_data_DataCollection
 * Method:    cSetValue
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_DataCollection_cSetValue
(JNIEnv *env, jobject thisObj, jlong dataCollectionAddr, jlong valueAddr, jlong idx)
{
    data_management::DataCollectionPtr pDataCollection =
        services::staticPointerCast<DataCollection, SerializationIface>(
            (*(data_management::SerializationIfacePtr *)dataCollectionAddr));
    (*(pDataCollection))[idx] = *((SerializationIfacePtr *)valueAddr);
}
