/* file: loss_forward_input.cpp */
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
#include "neural_networks/layers/loss/JLossForwardInput.h"
#include "neural_networks/layers/loss/JLossForwardInputId.h"

#include "daal.h"

#include "common_helpers.h"

#define dataId com_intel_daal_algorithms_neural_networks_layers_loss_LossForwardInputId_dataId
#define weightsId com_intel_daal_algorithms_neural_networks_layers_loss_LossForwardInputId_weightsId
#define biasesId com_intel_daal_algorithms_neural_networks_layers_loss_LossForwardInputId_biasesId
#define groundTruthId com_intel_daal_algorithms_neural_networks_layers_loss_LossForwardInputId_groundTruthId

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_loss_LossForwardInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_loss_LossForwardInput_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if (id == dataId || id == weightsId || id == biasesId || id == groundTruthId)
    {
        jniInput<loss::forward::Input>::set<loss::forward::InputId, Tensor>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_loss_LossForwardInput
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_loss_LossForwardInput_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if (id == dataId || id == weightsId || id == biasesId || id == groundTruthId)
    {
        return jniInput<loss::forward::Input>::get<loss::forward::InputId, Tensor>(inputAddr, id);
    }

    return (jlong)0;
}
