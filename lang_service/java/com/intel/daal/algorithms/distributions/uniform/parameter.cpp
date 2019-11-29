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
#include "com_intel_daal_algorithms_distributions_uniform_Parameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_distributions_uniform_Parameter
 * Method:    cSetA
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_distributions_uniform_Parameter_cSetA(JNIEnv * env, jobject thisObj, jlong cParameter,
                                                                                            jdouble a)
{
    (((distributions::uniform::Parameter<float> *)cParameter))->a = (float)a;
}

/*
 * Class:     com_intel_daal_algorithms_distributions_uniform_Parameter
 * Method:    cSetB
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_distributions_uniform_Parameter_cSetB(JNIEnv * env, jobject thisObj, jlong cParameter,
                                                                                            jdouble b)
{
    (((distributions::uniform::Parameter<float> *)cParameter))->b = (float)b;
}

/*
 * Class:     com_intel_daal_algorithms_distributions_uniform_Parameter
 * Method:    cGetA
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_distributions_uniform_Parameter_cGetA(JNIEnv * env, jobject thisObj, jlong cParameter)
{
    return (jdouble)((((distributions::uniform::Parameter<float> *)cParameter))->a);
}

/*
 * Class:     com_intel_daal_algorithms_distributions_uniform_Parameter
 * Method:    cGetB
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_distributions_uniform_Parameter_cGetB(JNIEnv * env, jobject thisObj, jlong cParameter)
{
    return (jdouble)((((distributions::uniform::Parameter<float> *)cParameter))->b);
}
