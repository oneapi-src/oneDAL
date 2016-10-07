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
#include "svm/JModel.h"
#include "daal.h"

using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms::svm;

/*
 * Class:     com_intel_daal_algorithms_svm_Model
 * Method:    cGetSupportVectors
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svm_Model_cGetSupportVectors
(JNIEnv *env, jobject obj, jlong modelAddr)
{
    NumericTablePtr *ptr = new NumericTablePtr();
    *ptr = (*(services::SharedPtr<Model> *)modelAddr)->getSupportVectors();

    return (jlong)ptr;
}

/*
 * Class:     com_intel_daal_algorithms_svm_Model
 * Method:    cGetClassificationCoefficients
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svm_Model_cGetClassificationCoefficients
(JNIEnv *env, jobject obj, jlong modelAddr)
{
    NumericTablePtr *ptr = new NumericTablePtr();
    *ptr = (*(services::SharedPtr<Model> *)modelAddr)->getClassificationCoefficients();

    return (jlong)ptr;
}

/*
 * Class:     com_intel_daal_algorithms_svm_Model
 * Method:    cGetBias
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_svm_Model_cGetBias
(JNIEnv *env, jobject obj, jlong modelAddr)
{
    return (jdouble)((*(services::SharedPtr<Model> *)modelAddr)->getBias());
}
