/* file: common_helpers_functions.h */
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

#ifndef __COMMON_HELPERS_FUNCTIONS_H__
#define __COMMON_HELPERS_FUNCTIONS_H__

#include <jni.h>
#include "daal.h"

namespace daal
{
using namespace daal::services;

jlongArray getJavaLongArrayFromSizeTCollection(JNIEnv *env, const Collection<size_t> &dims);
void throwError(JNIEnv *env, const char *message);
void checkError(JNIEnv *env, const services::Status& s);

}

#define DAAL_CHECK_THROW(s) daal::checkError(env, s)

#endif
