/* file: zlib_parameter.cpp */
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

#include "zlib/JZlibCompressionParameter.h"
#include "daal.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_compression_CompressionParameter
 * Method:    cInit
 * Signature:(I)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_compression_zlib_ZlibCompressionParameter_cSetGzHeader
(JNIEnv *env, jobject, jlong parAddr, jboolean header)
{
    (*((ZlibCompressionParameter *)parAddr)).gzHeader = header;
}

/*
* Class:     com_intel_daal_data_1management_compression_CompressionParameter
* Method:    cGetCompressionLevel
* Signature:(J)I
*/
JNIEXPORT jboolean JNICALL Java_com_intel_daal_data_1management_compression_zlib_ZlibCompressionParameter_cGetGzHeader
(JNIEnv *env, jobject, jlong parAddr)
{
    return(jboolean)(*((ZlibCompressionParameter *)parAddr)).gzHeader;
}
