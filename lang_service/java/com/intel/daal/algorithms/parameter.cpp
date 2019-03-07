/* file: parameter.cpp */
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
#include "JParameter.h"
#include "daal.h"

using namespace daal::services;

/*
 * Class:     com_intel_daal_algorithms_Parameter
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_Parameter_cDispose
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    delete(daal::algorithms::Parameter *)parAddr;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_Parameter_cCheck
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    ((daal::algorithms::Parameter *)parAddr)->check();
}

namespace daal
{

void throwError(JNIEnv *env, const char *message)
{
    env->ThrowNew(env->FindClass("java/lang/Exception"), message);
}

void checkError(JNIEnv *env, const services::Status& s)
{
    if(!s)
        throwError(env, s.getDescription());
}

}
