/* file: model.cpp */
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
    services::SharedPtr<algorithms::Model> *modelShPtr = (services::SharedPtr<algorithms::Model> *)modelAddr;
    delete modelShPtr;
}
