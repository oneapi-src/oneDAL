/* file: keyvaluedatacollection.cpp */
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

#include "JKeyValueDataCollection.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_management_data_KeyValueDataCollection
 * Method:    cNewDataCollection
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_KeyValueDataCollection_cNewDataCollection
  (JNIEnv *env, jobject thisObj)
{
    KeyValueDataCollection *collection = new KeyValueDataCollection();
    return (jlong)(new SerializationIfacePtr(collection));
}

/*
 * Class:     com_intel_daal_data_management_data_KeyValueDataCollection
 * Method:    cGetValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_KeyValueDataCollection_cGetValue
  (JNIEnv *env, jobject thisObj, jlong collectionAddr, jint key)
{
    SerializationIfacePtr *collectionShPtr = (SerializationIfacePtr *)collectionAddr;
    KeyValueDataCollection *collection = static_cast<KeyValueDataCollection *>(collectionShPtr->get());
    SerializationIfacePtr *value = new SerializationIfacePtr((*collection)[(size_t)key]);
    return (jlong)value;
}

/*
 * Class:     com_intel_daal_data_management_data_KeyValueDataCollection
 * Method:    cSetValue
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_KeyValueDataCollection_cSetValue
  (JNIEnv *env, jobject thisObj, jlong collectionAddr, jint key, jlong valueAddr)
{
    SerializationIfacePtr *collectionShPtr = (SerializationIfacePtr *)collectionAddr;
    SerializationIfacePtr *valueShPtr = (SerializationIfacePtr *)valueAddr;
    KeyValueDataCollection *collection = static_cast<KeyValueDataCollection *>(collectionShPtr->get());
    (*collection)[(size_t)key] = *valueShPtr;
}

/*
 * Class:     com_intel_daal_data_management_data_KeyValueDataCollection
 * Method:    cSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_KeyValueDataCollection_cSize
  (JNIEnv *env, jobject thisObj, jlong collectionAddr)
{
    SerializationIfacePtr *collectionShPtr = (SerializationIfacePtr *)collectionAddr;
    KeyValueDataCollection *collection = static_cast<KeyValueDataCollection *>(collectionShPtr->get());
    return (jlong)(collection->size());
}

/*
 * Class:     com_intel_daal_data_management_data_KeyValueDataCollection
 * Method:    cGetKeyByIndex
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_KeyValueDataCollection_cGetKeyByIndex
  (JNIEnv *env, jobject thisObj, jlong collectionAddr, jint index)
{
    SerializationIfacePtr *collectionShPtr = (SerializationIfacePtr *)collectionAddr;
    KeyValueDataCollection *collection = static_cast<KeyValueDataCollection *>(collectionShPtr->get());
    size_t key = collection->getKeyByIndex((size_t)index);
    return (jlong)(key);
}

/*
 * Class:     com_intel_daal_data_management_data_KeyValueDataCollection
 * Method:    cGetValueByIndex
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_KeyValueDataCollection_cGetValueByIndex
  (JNIEnv *env, jobject thisObj, jlong collectionAddr, jint index)
{
    SerializationIfacePtr *collectionShPtr = (SerializationIfacePtr *)collectionAddr;
    KeyValueDataCollection *collection = static_cast<KeyValueDataCollection *>(collectionShPtr->get());
    SerializationIfacePtr *valueShPtr = new SerializationIfacePtr(
        collection->getValueByIndex((size_t)index));
    return (jlong)valueShPtr;
}
