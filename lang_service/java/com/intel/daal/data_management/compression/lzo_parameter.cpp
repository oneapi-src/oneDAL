/* file: lzo_parameter.cpp */
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

#include "lzo/JLzoCompressionParameter.h"
#include "daal.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_compression_CompressionParameter
 * Method:    cInit
 * Signature:(I)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_compression_lzo_LzoCompressionParameter_cSetPreHeadBytes
(JNIEnv *env, jobject, jlong parAddr, jlong preBytes)
{
    (*((LzoCompressionParameter *)parAddr)).preHeadBytes = preBytes;
}

/*
* Class:     com_intel_daal_data_1management_compression_CompressionParameter
* Method:    cGetCompressionLevel
* Signature:(J)I
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_compression_lzo_LzoCompressionParameter_cGetPreHeadBytes
(JNIEnv *env, jobject, jlong parAddr)
{
    return(jlong)(*((LzoCompressionParameter *)parAddr)).preHeadBytes;
}

/*
 * Class:     com_intel_daal_data_1management_compression_CompressionParameter
 * Method:    cInit
 * Signature:(I)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_compression_lzo_LzoCompressionParameter_cSetPostHeadBytes
(JNIEnv *env, jobject, jlong parAddr, jlong postBytes)
{
    (*((LzoCompressionParameter *)parAddr)).postHeadBytes = postBytes;
}

/*
* Class:     com_intel_daal_data_1management_compression_CompressionParameter
* Method:    cGetCompressionLevel
* Signature:(J)I
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_compression_lzo_LzoCompressionParameter_cGetPostHeadBytes
(JNIEnv *env, jobject, jlong parAddr)
{
    return(jlong)(*((LzoCompressionParameter *)parAddr)).postHeadBytes;
}
