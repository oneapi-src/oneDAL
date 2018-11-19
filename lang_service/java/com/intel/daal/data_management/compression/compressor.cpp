/* file: compressor.cpp */
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

#include "JCompressor.h"
#include "daal.h"

using namespace daal;
using namespace daal::data_management;

#include "compression_types.i"

JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_compression_Compressor_cInit
(JNIEnv *env, jobject, jint method)
{
    jlong compressor = 0;
    switch(method)
    {
    case Zlib:
        compressor = (jlong)(new Compressor<data_management::zlib>());
        break;
    case Lzo:
        compressor = (jlong)(new Compressor<data_management::lzo>());
        break;
    case Rle:
        compressor = (jlong)(new Compressor<data_management::rle>());
        break;
    case Bzip2:
        compressor = (jlong)(new Compressor<data_management::bzip2>());
        break;
    default:
        break;
    }
    return compressor;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_compression_Compressor_cInitParameter
(JNIEnv *env, jobject, jlong comprAddr, jint method)
{
    jlong par = 0;
    switch(method)
    {
    case Zlib:
        par = (jlong) & (((Compressor<data_management::zlib> *)comprAddr)->parameter);
        break;
    case Lzo:
        par = (jlong) & (((Compressor<data_management::lzo> *)comprAddr)->parameter);
        break;
    case Rle:
        par = (jlong) & (((Compressor<data_management::rle> *)comprAddr)->parameter);
        break;
    case Bzip2:
        par = (jlong) & (((Compressor<data_management::bzip2> *)comprAddr)->parameter);
        break;
    default:
        break;
    }
    return par;
}
