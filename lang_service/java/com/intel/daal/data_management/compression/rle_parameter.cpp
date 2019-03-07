/* file: rle_parameter.cpp */
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
