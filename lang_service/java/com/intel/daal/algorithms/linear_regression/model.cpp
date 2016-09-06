/* file: model.cpp */
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
#include "linear_regression/JModel.h"
#include "daal.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::services;

/*
 * Class:     com_intel_daal_algorithms_linear_regression_Model
 * Method:    cGetBeta
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_Model_cGetBeta
(JNIEnv *env, jobject thisObj, jlong modAddr)
{
    NumericTablePtr *nt = new NumericTablePtr();
    SharedPtr<algorithms::linear_regression::Model> res = *(SharedPtr<algorithms::linear_regression::Model> *)modAddr;

    *nt = res->getBeta();

    return (jlong)nt;
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_Model
 * Method:    cGetNumberOfFeatures
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_Model_cGetNumberOfFeatures
(JNIEnv *env, jobject thisObj, jlong modAddr)
{
    SharedPtr<algorithms::linear_regression::Model> res = *(SharedPtr<algorithms::linear_regression::Model> *)modAddr;

    return (jlong)res->getNumberOfFeatures();
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_Model
 * Method:    cGetNumberOfBetas
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_Model_cGetNumberOfBetas
(JNIEnv *env, jobject thisObj, jlong modAddr)
{
    SharedPtr<algorithms::linear_regression::Model> res = *(SharedPtr<algorithms::linear_regression::Model> *)modAddr;

    return (jlong)res->getNumberOfBetas();
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_Model
 * Method:    cGetNumberOfResponses
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_Model_cGetNumberOfResponses
(JNIEnv *env, jobject thisObj, jlong modAddr)
{
    SharedPtr<algorithms::linear_regression::Model> res = *(SharedPtr<algorithms::linear_regression::Model> *)modAddr;

    return (jlong)res->getNumberOfResponses();
}
