/* file: modelnormeq.cpp */
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
#include "linear_regression/JModelNormEq.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_algorithms_linear_regression_ModelNormEq
 * Method:    cInitDouble
 * Signature: (JJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_ModelNormEq_cInitDouble
  (JNIEnv *env, jobject thisObj, jlong nFeatures, jlong nResponses, jlong parAddr)
{
    double dummy = 0.0;
    linear_regression::Parameter *par = (linear_regression::Parameter *)parAddr;
    linear_regression::ModelNormEq *modelPtr =
        new linear_regression::ModelNormEq((size_t)nFeatures, (size_t)nResponses, *par, dummy);
    services::SharedPtr<linear_regression::ModelNormEq> *modelShPtr = new services::SharedPtr<linear_regression::ModelNormEq>(modelPtr);
    return (jlong)modelShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_ModelNormEq
 * Method:    cInitFloat
 * Signature: (JJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_ModelNormEq_cInitFloat
  (JNIEnv *env, jobject thisObj, jlong nFeatures, jlong nResponses, jlong parAddr)
{
    float dummy = 0.0f;
    linear_regression::Parameter *par = (linear_regression::Parameter *)parAddr;
    linear_regression::ModelNormEq *modelPtr =
        new linear_regression::ModelNormEq((size_t)nFeatures, (size_t)nResponses, *par, dummy);
    services::SharedPtr<linear_regression::ModelNormEq> *modelShPtr = new services::SharedPtr<linear_regression::ModelNormEq>(modelPtr);
    return (jlong)modelShPtr;
}
