/* file: data_collection.cpp */
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
