/* file: distributed_step2_master_input.cpp */
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
#include "neural_networks/training/JDistributedStep2MasterInput.h"
#include "neural_networks/training/JDistributedStep2MasterInputId.h"

#include "daal.h"
#include "common_helpers.h"

#define partialResultsId com_intel_daal_algorithms_neural_networks_training_DistributedStep2MasterInputId_partialResultsId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep2MasterInput
 * Method:    cAddInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep2MasterInput_cAddInput
  (JNIEnv *, jobject, jlong inputAddr, jint id, jint key, jlong partialResultAddr)
{
    jniInput<training::DistributedInput<step2Master> >::add<training::Step2MasterInputId, training::PartialResult>(inputAddr, id, key, partialResultAddr);

}
