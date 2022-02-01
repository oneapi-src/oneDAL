/* file: model.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "daal.h"
#include "com_intel_daal_algorithms_multinomial_naive_bayes_Model.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_Model_cGetLogP(JNIEnv * env, jobject thisObj, jlong modelAddr)
{
    NumericTablePtr * nt                  = new NumericTablePtr();
    multinomial_naive_bayes::ModelPtr res = *(multinomial_naive_bayes::ModelPtr *)modelAddr;

    *nt = res->getLogP();

    return (jlong)nt;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_Model_cGetLogTheta(JNIEnv * env, jobject thisObj, jlong modelAddr)
{
    NumericTablePtr * nt                  = new NumericTablePtr();
    multinomial_naive_bayes::ModelPtr res = *(multinomial_naive_bayes::ModelPtr *)modelAddr;

    *nt = res->getLogTheta();

    return (jlong)nt;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_Model_cGetNFeatures(JNIEnv * env, jobject thisObj, jlong modelAddr)
{
    multinomial_naive_bayes::ModelPtr res = *(multinomial_naive_bayes::ModelPtr *)modelAddr;

    return (jlong)(res->getNFeatures());
}
