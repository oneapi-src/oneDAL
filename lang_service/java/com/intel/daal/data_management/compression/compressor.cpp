/* file: compressor.cpp */
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
