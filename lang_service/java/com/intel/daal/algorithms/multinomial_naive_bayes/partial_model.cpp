/* file: partial_model.cpp */
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

#include "daal.h"
#include "multinomial_naive_bayes/JPartialModel.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_PartialModel_cGetClassGroupSum
(JNIEnv *env, jobject thisObj, jlong modelAddr)
{
    NumericTablePtr *nt = new NumericTablePtr();
    services::SharedPtr<multinomial_naive_bayes::PartialModel> res = *(services::SharedPtr<multinomial_naive_bayes::PartialModel> *)modelAddr;

    *nt = res->getClassGroupSum();

    return (jlong)nt;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_PartialModel_cGetClassSize
(JNIEnv *env, jobject thisObj, jlong modelAddr)
{
    NumericTablePtr *nt = new NumericTablePtr();
    services::SharedPtr<multinomial_naive_bayes::PartialModel> res = *(services::SharedPtr<multinomial_naive_bayes::PartialModel> *)modelAddr;

    *nt = res->getClassSize();

    return (jlong)nt;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_PartialModel_cGetNFeatures
(JNIEnv *env, jobject thisObj, jlong modelAddr)
{
    services::SharedPtr<multinomial_naive_bayes::PartialModel> res = *(services::SharedPtr<multinomial_naive_bayes::PartialModel> *)modelAddr;

    return (jlong)(res->getNFeatures());
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_PartialModel_cGetNObservations
(JNIEnv *env, jobject thisObj, jlong modelAddr)
{
    services::SharedPtr<multinomial_naive_bayes::PartialModel> res = *(services::SharedPtr<multinomial_naive_bayes::PartialModel> *)modelAddr;

    return (jlong)(res->getNObservations());
}
