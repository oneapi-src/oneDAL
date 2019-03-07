/* file: compression_parameter.cpp */
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

#include "JCompressionParameter.h"
#include "daal.h"

using namespace daal;
using namespace daal::data_management;

#include "compression_types.i"

/*
 * Class:     com_intel_daal_data_1management_compression_CompressionParameter
 * Method:    cInit
 * Signature:(I)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_compression_CompressionParameter_cSetCompressionLevel
(JNIEnv *env, jobject, jlong parAddr, jint cLevel)
{
    switch(cLevel)
    {
    case DefaultLevel:
        (*((CompressionParameter *)parAddr)).level = defaultLevel;
        break;
    case Level0:
        (*((CompressionParameter *)parAddr)).level = level0;
        break;
    case Level1:
        (*((CompressionParameter *)parAddr)).level = level1;
        break;
    case Level2:
        (*((CompressionParameter *)parAddr)).level = level2;
        break;
    case Level3:
        (*((CompressionParameter *)parAddr)).level = level3;
        break;
    case Level4:
        (*((CompressionParameter *)parAddr)).level = level4;
        break;
    case Level5:
        (*((CompressionParameter *)parAddr)).level = level5;
        break;
    case Level6:
        (*((CompressionParameter *)parAddr)).level = level6;
        break;
    case Level7:
        (*((CompressionParameter *)parAddr)).level = level7;
        break;
    case Level8:
        (*((CompressionParameter *)parAddr)).level = level8;
        break;
    case Level9:
        (*((CompressionParameter *)parAddr)).level = level9;
        break;
    default:
        break;
    }
}

/*
* Class:     com_intel_daal_data_1management_compression_CompressionParameter
* Method:    cGetCompressionLevel
* Signature:(J)I
*/
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_compression_CompressionParameter_cGetCompressionLevel
(JNIEnv *env, jobject, jlong parameterAddress)
{
    CompressionLevel level = ((CompressionParameter *)parameterAddress)->level;
    switch(level)
    {
    case defaultLevel:
        return DefaultLevel;
    case level0:
        return Level0;
    case level1:
        return Level1;
    case level2:
        return Level2;
    case level3:
        return Level3;
    case level4:
        return Level4;
    case level5:
        return Level5;
    case level6:
        return Level6;
    case level7:
        return Level7;
    case level8:
        return Level8;
    case level9:
        return Level9;
    default:
        return DefaultLevel;
    }
}
