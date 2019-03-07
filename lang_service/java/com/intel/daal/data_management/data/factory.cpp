/* file: factory.cpp */
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

#include "JFactory.h"

#include "java_numeric_table.h"
#include "java_tensor.h"

using namespace daal;
using namespace daal::data_management;
using namespace daal::services;

/*
 * Class:     com_intel_daal_data_management_data_Factory
 * Method:    cGetSerializationTag
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_Factory_cGetSerializationTag
  (JNIEnv *env, jobject thisObj, jlong serializableAddr)
{
    SerializationIfacePtr *object = (SerializationIfacePtr *)serializableAddr;
    int tag = (*object)->getSerializationTag();
    return (jint)tag;
}

/*
 * Class:     com_intel_daal_data_management_data_Factory
 * Method:    cGetJavaNumericTable
 * Signature: (OJ)O
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_Factory_cGetJavaNumericTable
  (JNIEnv *env, jobject thisObj, jlong cObject)
{
    SerializationIfacePtr *object = (SerializationIfacePtr *)cObject;
    JavaNumericTableBase *nt = dynamic_cast<JavaNumericTableBase*>(object->get());

    if (nt != 0) { return nt->getJavaObject(); }

    return 0;
}

/*
 * Class:     com_intel_daal_data_management_data_Factory
 * Method:    cGetJavaTensor
 * Signature: (OJ)O
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_Factory_cGetJavaTensor
  (JNIEnv *env, jobject thisObj, jlong cObject)
{
    SerializationIfacePtr *object = (SerializationIfacePtr *)cObject;
    JavaTensorBase *nt = dynamic_cast<JavaTensorBase*>(object->get());

    if (nt != 0) { return nt->getJavaObject(); }

    return 0;
}
