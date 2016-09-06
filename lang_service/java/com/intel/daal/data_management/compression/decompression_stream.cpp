/* file: decompression_stream.cpp */
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

#include "JDecompressionStream.h"
#include "daal.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_compression_Decompression
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_compression_DecompressionStream_cDispose
(JNIEnv *env, jobject, jlong strAddr)
{
    delete(DecompressionStream *)strAddr;
}

/*
 * Class:     com_intel_daal_data_1management_compression_Decompression
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_compression_DecompressionStream_cInit
(JNIEnv *env, jobject, jlong comprAddr, jlong minSize)
{
    jlong strmAddr = 0;
    strmAddr = (jlong)(new DecompressionStream((DecompressorImpl *)comprAddr, minSize));

    if(((DecompressionStream *)strmAddr)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"),
                      ((DecompressionStream *)strmAddr)->getErrors()->getDescription());
    }
    return strmAddr;
}

/*
 * Class:     com_intel_daal_data_1management_compression_Decompression
 * Method:    cSetInputDataBlock
 * Signature:(J[BJJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_compression_DecompressionStream_cAdd
(JNIEnv *env, jobject, jlong strmAddr, jbyteArray inBlock, jlong size)
{
    jbyte *inBuffer = env->GetByteArrayElements(inBlock, 0);

    DataBlock tmp((byte *)inBuffer, (size_t)size);

    ((CompressionStream *)strmAddr)->push_back(&tmp);

    if(((DecompressionStream *)strmAddr)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"),
                      ((DecompressionStream *)strmAddr)->getErrors()->getDescription());
    }
}

/*
 * Class:     com_intel_daal_data_1management_compression_Decompression
 * Method:    cIsOutputDataBlockFull
 * Signature:(J)Z
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_compression_DecompressionStream_cGetDecompressedDataSize
(JNIEnv *env, jobject, jlong strmAddr)
{

    if(((DecompressionStream *)strmAddr)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"),
                      ((DecompressionStream *)strmAddr)->getErrors()->getDescription());
    }
    return(jlong)((DecompressionStream *)strmAddr)->getDecompressedDataSize();
}

/*
 * Class:     com_intel_daal_data_1management_compression_Decompression
 * Method:    cRun
 * Signature:(J[BJJ)I
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_compression_DecompressionStream_cCopyDecompressedArray
(JNIEnv *env, jobject, jlong strmAddr, jbyteArray outBlock, jlong chunkSize)
{
    jbyte *outBuffer = env->GetByteArrayElements(outBlock, 0);
    jlong result = (jlong)((DecompressionStream *)strmAddr)->copyDecompressedArray((byte *)outBuffer, (size_t)chunkSize);
    env->ReleaseByteArrayElements(outBlock, outBuffer, 0);

    if(((DecompressionStream *)strmAddr)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"),
                      ((DecompressionStream *)strmAddr)->getErrors()->getDescription());
    }

    return result;
}
