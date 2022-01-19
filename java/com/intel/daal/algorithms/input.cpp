/* file: input.cpp */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_Result */

#include "daal.h"
#include "com_intel_daal_algorithms_Input.h"

/*
 * Class:     com_intel_daal_algorithms_Result
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_Input_cCheck(JNIEnv * env, jobject thisObj, jlong inputAddr, jlong parAddr, jint method)
{
    ((daal::algorithms::Input *)inputAddr)->check((daal::algorithms::Parameter *)parAddr, method);
}
