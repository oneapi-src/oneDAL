/* file: compression.cpp */
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

#include "JCompression.h"
#include "daal.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_compression_Compression
 * Method:    cIsOutputDataBlockFull
 * Signature:(J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_data_1management_compression_Compression_cIsOutputDataBlockFull
(JNIEnv *env, jobject, jlong compressionAddress)
{
    using namespace daal;

    if(((Compression *)compressionAddress)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"),
                      ((Compression *)compressionAddress)->getErrors()->getDescription());
    }
    return((Compression *)compressionAddress)->isOutputDataBlockFull();
}

/*
 * Class:     com_intel_daal_data_1management_compression_Compression
 * Method:    cGetUsedOutputDataBlockSize
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_compression_Compression_cGetUsedOutputDataBlockSize
(JNIEnv *env, jobject, jlong compressionAddress)
{
    using namespace daal;

    if(((Compression *)compressionAddress)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"),
                      ((Compression *)compressionAddress)->getErrors()->getDescription());
    }
    return((Compression *)compressionAddress)->getUsedOutputDataBlockSize();
}

/*
 * Class:     com_intel_daal_data_1management_compression_Compression
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_compression_Compression_cDispose
(JNIEnv *env, jobject, jlong compressionAddress)
{
    using namespace daal;
    delete(Compression *)compressionAddress;
}

/*
 * Class:     com_intel_daal_data_1management_compression_Compression
 * Method:    cSetInputDataBlock
 * Signature:(J[BJJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_compression_Compression_cSetInputDataBlock
(JNIEnv *env, jobject, jlong compressionAddress, jbyteArray inStream, jlong size, jlong offset)
{
    using namespace daal;

    jbyte *inStreamBuffer = env->GetByteArrayElements(inStream, 0);

    ((Compression *)compressionAddress)->setInputDataBlock((byte *)inStreamBuffer, size, offset);

    if(((Compression *)compressionAddress)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"),
                      ((Compression *)compressionAddress)->getErrors()->getDescription());
    }
}

/*
 * Class:     com_intel_daal_data_1management_compression_Compression
 * Method:    cRun
 * Signature:(J[BJJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_compression_Compression_cRun
(JNIEnv *env, jobject, jlong compressionAddress, jbyteArray outStream, jlong chunkSize, jlong offset)
{
    using namespace daal;

    jbyte *outStreamBuffer = env->GetByteArrayElements(outStream, 0);
    ((Compression *)compressionAddress)->run((byte *)outStreamBuffer, (size_t)chunkSize, (size_t)offset);
    env->ReleaseByteArrayElements(outStream, outStreamBuffer, 0);

    if(((Compression *)compressionAddress)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"),
                      ((Compression *)compressionAddress)->getErrors()->getDescription());
    }
}

/*
 * Class:     com_intel_daal_data_1management_compression_Compression
 * Method:    cCheckInputParams
 * Signature:(J[BJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_compression_Compression_cCheckInputParams
(JNIEnv *env, jobject, jlong compressionAddress, jbyteArray inStream, jlong size)
{
    using namespace daal;

    jbyte *inStreamBuffer = env->GetByteArrayElements(inStream, 0);

    ((Compression *)compressionAddress)->checkInputParams((byte *)inStreamBuffer, (size_t)size);

    if(((Compression *)compressionAddress)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"),
                      ((Compression *)compressionAddress)->getErrors()->getDescription());
    }
}

/*
 * Class:     com_intel_daal_data_1management_compression_Compression
 * Method:    cCheckOutputParams
 * Signature:(J[BJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_compression_Compression_cCheckOutputParams
(JNIEnv *env, jobject, jlong compressionAddress, jbyteArray outStream, jlong size)
{
    using namespace daal;

    jbyte *outStreamBuffer = env->GetByteArrayElements(outStream, 0);

    ((Compression *)compressionAddress)->checkOutputParams((byte *)outStreamBuffer, (size_t)size);

    if(((Compression *)compressionAddress)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"),
                      ((Compression *)compressionAddress)->getErrors()->getDescription());
    }
}
