/* file: distributed_partial_result_step6.cpp */
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
#include "daal.h"
#include "com_intel_daal_algorithms_gbt_regression_training_DistributedPartialResultStep6.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::gbt::regression::training;

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_training_DistributedPartialResultStep6_cNewDistributedPartialResultStep6
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<DistributedPartialResultStep6>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_gbt_regression_training_DistributedPartialResultStep6
 * Method:    cGetModel
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_training_DistributedPartialResultStep6_cGetModel
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    ModelPtr *m = new ModelPtr();
    DistributedPartialResultStep6Ptr res = services::staticPointerCast<DistributedPartialResultStep6, SerializationIface>(*((SerializationIfacePtr *)resAddr));
    *m = res->get(partialModel);
    return (jlong)m;
}
