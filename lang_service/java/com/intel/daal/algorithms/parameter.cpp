/* file: parameter.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#include "com_intel_daal_algorithms_Parameter.h"
#include "daal.h"

using namespace daal::services;

/*
 * Class:     com_intel_daal_algorithms_Parameter
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_Parameter_cDispose(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    delete (daal::algorithms::Parameter *)parAddr;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_Parameter_cCheck(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    ((daal::algorithms::Parameter *)parAddr)->check();
}

namespace daal
{
void throwError(JNIEnv * env, const char * message)
{
    env->ThrowNew(env->FindClass("java/lang/Exception"), message);
}

void checkError(JNIEnv * env, const services::Status & s)
{
    if (!s) throwError(env, s.getDescription());
}

} // namespace daal
