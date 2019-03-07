/* file: decompressor.cpp */
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

#include "JDecompressor.h"
#include "daal.h"

using namespace daal;
using namespace daal::data_management;

#include "compression_types.i"

JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_compression_Decompressor_cInit
(JNIEnv *env, jobject, jint method)
{
    jlong decompressor = 0;
    switch(method)
    {
    case Zlib:
        decompressor = (jlong)(new Decompressor<data_management::zlib>());
        break;
    case Lzo:
        decompressor = (jlong)(new Decompressor<data_management::lzo>());
        break;
    case Rle:
        decompressor = (jlong)(new Decompressor<data_management::rle>());
        break;
    case Bzip2:
        decompressor = (jlong)(new Decompressor<data_management::bzip2>());
        break;
    default:
        break;
    }
    return decompressor;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_compression_Decompressor_cInitParameter
(JNIEnv *env, jobject, jlong comprAddr, jint method)
{
    jlong par = 0;
    switch(method)
    {
    case Zlib:
        par = (jlong) & (((Decompressor<data_management::zlib> *)comprAddr)->parameter);
        break;
    case Lzo:
        par = (jlong) & (((Decompressor<data_management::lzo> *)comprAddr)->parameter);
        break;
    case Rle:
        par = (jlong) & (((Decompressor<data_management::rle> *)comprAddr)->parameter);
        break;
    case Bzip2:
        par = (jlong) & (((Decompressor<data_management::bzip2> *)comprAddr)->parameter);
        break;
    default:
        break;
    }
    return par;
}
