/* file: distributed_partial_result.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
#include "neural_networks/training/JDistributedPartialResult.h"
#include "neural_networks/training/JDistributedPartialResultId.h"

#include "daal.h"
#include "common_helpers.h"

#define resultFromMasterId com_intel_daal_algorithms_neural_networks_training_DistributedPartialResultId_resultFromMasterId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedPartialResult
 * Method:    cNewPartialResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedPartialResult_cNewPartialResult
  (JNIEnv *, jobject)
{
    return jniArgument<training::DistributedPartialResult>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedPartialResult
 * Method:    cGetResult
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedPartialResult_cGetResult
  (JNIEnv *, jobject, jlong algAddr, jint id)
{
    if (id == resultFromMasterId)
    {
        return jniArgument<training::DistributedPartialResult>::get<training::Step2MasterPartialResultId, training::Result>(algAddr, id);
    }

    return (jlong)0;
}
