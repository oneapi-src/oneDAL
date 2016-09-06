/* file: prediction_input.cpp */
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
#include "neural_networks/prediction/JPredictionInput.h"
#include "neural_networks/prediction/JPredictionTensorInputId.h"
#include "neural_networks/prediction/JPredictionModelInputId.h"

#include "daal.h"

#include "common_helpers.h"

#define dataId com_intel_daal_algorithms_neural_networks_prediction_PredictionTensorInputId_dataId
#define modelId com_intel_daal_algorithms_neural_networks_prediction_PredictionModelInputId_modelId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionInput_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if (id == dataId)
    {
        jniInput<prediction::Input>::set<prediction::TensorInputId, Tensor>(inputAddr, id, ntAddr);
    } else if (id == modelId)
    {
        jniInput<prediction::Input>::set<prediction::ModelInputId, prediction::Model>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionInput
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionInput_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if (id == dataId)
    {
        return jniInput<prediction::Input>::get<prediction::TensorInputId, Tensor>(inputAddr, id);
    } else if (id == modelId)
    {
        return jniInput<prediction::Input>::get<prediction::ModelInputId, prediction::Model>(inputAddr, id);
    }

    return (jlong)0;
}
