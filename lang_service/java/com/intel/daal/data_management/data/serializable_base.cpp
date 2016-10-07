/* file: serializable_base.cpp */
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

#include "JSerializableBase.h"
#include "daal.h"

#include "java_numeric_table.h"

using namespace daal::data_management;
using namespace daal::services;

/*
 * Class:     com_intel_daal_data_1management_data_SerializableBase
 * Method:    cSerializeCObject
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobjectArray JNICALL Java_com_intel_daal_data_1management_data_SerializableBase_cSerializeCObject
(JNIEnv *env, jobject thisObj, jlong ptr)
{
    SerializationIface *nt = (*(SerializationIfacePtr *)ptr).get();
    InputDataArchive dataArch;
    nt->serialize(dataArch);

    size_t length = dataArch.getSizeOfArchive();

    size_t maxBlockLenght = 1024*1024*1024;
    size_t nBlocks = length/maxBlockLenght;
    if(  length != nBlocks*maxBlockLenght ) nBlocks++;

    daal::byte *buffer = (daal::byte *)daal_malloc(length);
    if (!buffer)
    {
        Error e(ErrorMemoryAllocationFailed);
        const char *description = e.description();
        env->ThrowNew(env->FindClass("java/lang/Exception"), description);
        return NULL;
    }
    dataArch.copyArchiveToArray(buffer, length);

    if(dataArch.getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), dataArch.getErrors()->getDescription());
        daal_free(buffer);
        return NULL;
    }

    jclass byteArrayClass = env->FindClass("[B");

    jobjectArray byte2dArray = env->NewObjectArray((jsize)nBlocks, byteArrayClass, NULL);

    size_t offset=0;
    for(size_t i=0; i<nBlocks; i++)
    {
        int smallLength = maxBlockLenght;
        if(i==nBlocks-1)
        {
            smallLength = length - i*maxBlockLenght;
        }

        jbyteArray byteArray = env->NewByteArray(smallLength);
        env->SetByteArrayRegion(byteArray, 0, smallLength, ((jbyte*)buffer)+offset);
        env->SetObjectArrayElement(byte2dArray, (jsize)i, byteArray);
        env->DeleteLocalRef(byteArray);

        offset += smallLength;
    }

    daal_free(buffer);

    return byte2dArray;
}

/*
 * Class:     com_intel_daal_data_1management_data_SerializableBase
 * Method:    cDeserializeCObject
 * Signature: (Ljava/nio/ByteBuffer;J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_SerializableBase_cDeserializeCObject
(JNIEnv *env, jobject thisObj, jobjectArray byte2dArray)
{
    int length2D = env->GetArrayLength(byte2dArray);

    size_t length = 0;
    for(int i=0; i<length2D; i++)
    {
        jbyteArray byteArray = (jbyteArray)env->GetObjectArrayElement(byte2dArray, (jsize)i);
        length += env->GetArrayLength(byteArray);
        env->DeleteLocalRef(byteArray);
    }

    daal::byte *buffer = (daal::byte *)daal_malloc(length);
    if (!buffer)
    {
        Error e(ErrorMemoryAllocationFailed);
        const char *description = e.description();
        env->ThrowNew(env->FindClass("java/lang/Exception"), description);
        return (jlong)0;
    }

    size_t offset = 0;
    for(int i=0; i<length2D; i++)
    {
        jbyteArray byteArray = (jbyteArray)env->GetObjectArrayElement(byte2dArray, (jsize)i);
        int smallLength = env->GetArrayLength(byteArray);
        env->GetByteArrayRegion(byteArray, 0, smallLength, ((jbyte*)buffer)+offset);
        env->DeleteLocalRef(byteArray);

        offset += smallLength;
    }

    OutputDataArchive dataArch(buffer, offset);

    SerializationIfacePtr *sPtr = new SerializationIfacePtr();

    *sPtr = dataArch.getAsSharedPtr();

    if(dataArch.getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), dataArch.getErrors()->getDescription());
        daal_free(buffer);
        return (jlong)0;
    }

    daal_free(buffer);

    return (jlong)sPtr;
}

/*
 * Class:     com_intel_daal_data_1management_data_SerializableBase
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_SerializableBase_cDispose
(JNIEnv *env, jobject thisObj, jlong ptr)
{
    delete (SerializationIfacePtr *)ptr;
}

JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_SerializableBase_throwUnpacked
(JNIEnv *env, jobject thisObj)
{
    env->ThrowNew(env->FindClass("java/lang/Exception"), "Object should be unpacked before further usage");
}

/*
 * Class:     com_intel_daal_data_1management_data_SerializableBase
 * Method:    cSetJavaVM
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_SerializableBase_cSetJavaVM
(JNIEnv *env, jobject thisObj)
{
    JavaVM *jvm;
    jint status = env->GetJavaVM(&jvm);
    if(status != 0)
    {
        return;
    }
    daal::JavaNumericTable::setJavaVM(jvm);
}

/*
 * Class:     com_intel_daal_data_1management_data_SerializableBase
 * Method:    cSetDaalContext
 * Signature: (Ljava/com/intel/daal/services/DaalContext)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_SerializableBase_cSetDaalContext
(JNIEnv *env, jobject thisObj, jobject context)
{
    daal::JavaNumericTable::setDaalContext(env->NewGlobalRef(context));
}

/*
 * Class:     com_intel_daal_data_1management_data_SerializableBase
 * Method:    cClearDaalContext
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_SerializableBase_cClearDaalContext
(JNIEnv *env, jobject thisObj)
{
    env->DeleteGlobalRef(daal::JavaNumericTable::getDaalContext());
    daal::JavaNumericTable::setDaalContext(NULL);
}
