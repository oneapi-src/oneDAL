/* file: lzo_parameter.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
