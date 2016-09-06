/* file: kernelfunction_rbf_parameter.cpp */
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
#include "rbf/JParameter.h"
#include "daal.h"

using namespace daal::algorithms::kernel_function::rbf;
/*
 * Class:     com_intel_daal_algorithms_kernel_function_rbf_Parameter
 * Method:    cSetSigma
 * Signature:(D)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kernel_1function_rbf_Parameter_cSetSigma
(JNIEnv *env, jobject thisObj, jlong parAddr, jdouble sigma)
{
    (*(Parameter *)parAddr).sigma = sigma;
}
/*
 * Class:     com_intel_daal_algorithms_kernel_function_rbf_Parameter
 * Method:    cGetSigma
 * Signature:(D)J
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Parameter_cGetSigma
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return(jdouble)(*(Parameter *)parAddr).sigma;
}
