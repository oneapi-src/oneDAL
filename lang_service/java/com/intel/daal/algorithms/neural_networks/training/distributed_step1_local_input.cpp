/* file: distributed_step1_local_input.cpp */
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
#include "neural_networks/training/JDistributedStep1LocalInput.h"
#include "neural_networks/training/JDistributedStep1LocalInputId.h"

#include "daal.h"
#include "common_helpers.h"

#define dataId com_intel_daal_algorithms_neural_networks_training_TrainingInputId_dataId
#define groundTruthId com_intel_daal_algorithms_neural_networks_training_TrainingInputId_groundTruthId
#define inputModelId com_intel_daal_algorithms_neural_networks_training_DistributedStep1LocalInputId_inputModelId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep1LocalInput
 * Method:    cSetModel
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep1LocalInput_cSetModel
  (JNIEnv *, jobject, jlong inputAddr, jint id, jlong modelAddr)
{
    if (id == inputModelId)
    {
        jniInput<training::DistributedInput<step1Local> >::set<training::Step1LocalInputId, training::Model>(inputAddr, id, modelAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep1LocalInput
 * Method:    cGetModel
 * Signature: (JIJ)V
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep1LocalInput_cGetModel
  (JNIEnv *, jobject, jlong inputAddr, jint id)
{
    if (id == inputModelId)
    {
        return jniInput<training::DistributedInput<step1Local> >::get<training::Step1LocalInputId, training::Model>(inputAddr, id);
    }
    return (jlong)0;
}
