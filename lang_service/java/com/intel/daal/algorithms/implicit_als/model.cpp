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

#include "daal.h"

#include "implicit_als/JModel.h"

using namespace daal;
using namespace daal::algorithms::implicit_als;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Model
 * Method:    cGetUsersFactors
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_Model_cGetUsersFactors
(JNIEnv *env, jobject thisObj, jlong modAddr)
{
    NumericTablePtr *nt = new NumericTablePtr();
    services::SharedPtr<algorithms::implicit_als::Model> res = *(services::SharedPtr<algorithms::implicit_als::Model> *)modAddr;

    *nt = res->getUsersFactors();

    return (jlong)nt;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Model
 * Method:    cGetItemsFactors
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_Model_cGetItemsFactors
(JNIEnv *env, jobject thisObj, jlong modAddr)
{
    NumericTablePtr *nt = new NumericTablePtr();
    services::SharedPtr<algorithms::implicit_als::Model> res = *(services::SharedPtr<algorithms::implicit_als::Model> *)modAddr;

    *nt = res->getItemsFactors();

    return (jlong)nt;
}
