/* file: rle_parameter.cpp */
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

#include "rle/JRleCompressionParameter.h"
#include "daal.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_compression_CompressionParameter
 * Method:    cInit
 * Signature:(I)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_compression_rle_RleCompressionParameter_cSetBlockHeader
(JNIEnv *env, jobject, jlong parAddr, jboolean header)
{
    (*((RleCompressionParameter *)parAddr)).isBlockHeader = header;
}

/*
* Class:     com_intel_daal_data_1management_compression_CompressionParameter
* Method:    cGetCompressionLevel
* Signature:(J)I
*/
JNIEXPORT jboolean JNICALL Java_com_intel_daal_data_1management_compression_rle_RleCompressionParameter_cGetBlockHeader
(JNIEnv *env, jobject, jlong parAddr)
{
    return(jboolean)(*((RleCompressionParameter *)parAddr)).isBlockHeader;
}
