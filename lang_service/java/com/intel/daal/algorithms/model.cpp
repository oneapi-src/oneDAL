/* file: model.cpp */
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
#include "JModel.h"
#include "daal.h"

using namespace daal;
using namespace daal::services;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_algorithms_Model
 * Method:    cSerializeCObject
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_algorithms_Model_cSerializeCObject
(JNIEnv *env, jobject thisObj, jlong modelPtr)
{
    SerializationIface *model = ((SerializationIfacePtr *)modelPtr)->get();
    InputDataArchive dataArch;
    model->serialize(dataArch);

    size_t length = dataArch.getSizeOfArchive();

    byte *buffer = (byte *)daal_malloc(length);
    dataArch.copyArchiveToArray(buffer, length);

    return env->NewDirectByteBuffer(buffer, length);
}

/*
 * Class:     com_intel_daal_algorithms_Model
 * Method:    cFreeByteBuffer
 * Signature: (Ljava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_Model_cFreeByteBuffer
(JNIEnv *env, jobject thisObj, jobject byteBuffer)
{
    byte *buffer = (byte *)(env->GetDirectBufferAddress(byteBuffer));
    daal_free(buffer);
}

/*
 * Class:     com_intel_dal_algorithms_Model
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_Model_cDispose
(JNIEnv *env, jobject thisObj, jlong modelAddr)
{
    algorithms::ModelPtr *modelShPtr = (algorithms::ModelPtr *)modelAddr;
    delete modelShPtr;
}
