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
#include "com_intel_daal_algorithms_distributions_bernoulli_Parameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_distributions_bernoulli_Parameter
 * Method:    cSetP
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_distributions_bernoulli_Parameter_cSetP(JNIEnv * env, jobject thisObj, jlong cParameter,
                                                                                              jdouble p)
{
    (((distributions::bernoulli::Parameter<float> *)cParameter))->p = (double)p;
}

/*
 * Class:     com_intel_daal_algorithms_distributions_bernoulli_Parameter
 * Method:    cGetP
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_distributions_bernoulli_Parameter_cGetP(JNIEnv * env, jobject thisObj, jlong cParameter)
{
    return (jdouble)((((distributions::bernoulli::Parameter<float> *)cParameter))->p);
}
