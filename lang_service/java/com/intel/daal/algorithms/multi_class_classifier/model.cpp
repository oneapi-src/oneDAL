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
#include "multi_class_classifier/JModel.h"
#include "daal.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_Model
 * Method:    cGetNumberOfTwoClassClassifierModels
 * Signature:(JJ)I
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_Model_cGetNumberOfTwoClassClassifierModels
(JNIEnv *env, jobject thisObj, jlong modAddr)
{
    services::SharedPtr<multi_class_classifier::Model> models = *(services::SharedPtr<multi_class_classifier::Model> *)modAddr;
    return (jlong)models->getNumberOfTwoClassClassifierModels();
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_Model
 * Method:    cGetTwoClassClassifierModel
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_Model_cGetTwoClassClassifierModel
(JNIEnv *env, jobject thisObj, jlong modAddr, jlong idx)
{
    services::SharedPtr<multi_class_classifier::Model> models = *(services::SharedPtr<multi_class_classifier::Model> *)modAddr;
    SerializationIfacePtr * dShPtr = new SerializationIfacePtr();
    *dShPtr = models->getTwoClassClassifierModel(idx);
    return (jlong)dShPtr;
}
