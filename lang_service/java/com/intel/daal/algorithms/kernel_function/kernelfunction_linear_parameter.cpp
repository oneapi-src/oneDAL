/* file: kernelfunction_linear_parameter.cpp */
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
#include "linear/JParameter.h"
#include "daal.h"

using namespace daal::algorithms::kernel_function::linear;
/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Parameter
 * Method:    cSetK
 * Signature:(DD)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Parameter_cSetK
(JNIEnv *env, jobject thisObj, jlong parAddr, jdouble k)
{
    (*(Parameter *)parAddr).k = k;
}
/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Parameter
 * Method:    cSetK
 * Signature:(DD)J
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Parameter_cGetK
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return(jdouble)(*(Parameter *)parAddr).k;
}

/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Parameter
 * Method:    cSetK
 * Signature:(DD)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Parameter_cSetB
(JNIEnv *env, jobject thisObj, jlong parAddr, jdouble b)
{
    (*(Parameter *)parAddr).b = b;
}
/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Parameter
 * Method:    cGetB
 * Signature:(DD)J
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Parameter_cGetB
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return(jdouble)(*(Parameter *)parAddr).b;
}
