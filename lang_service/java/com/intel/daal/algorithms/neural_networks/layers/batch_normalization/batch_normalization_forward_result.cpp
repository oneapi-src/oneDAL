/* file: batch_normalization_forward_result.cpp */
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
#include "neural_networks/layers/batch_normalization/JBatchNormalizationForwardResult.h"
#include "neural_networks/layers/batch_normalization/JBatchNormalizationLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define auxDataId com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationLayerDataId_auxDataId
#define auxWeightsId com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationLayerDataId_auxWeightsId
#define auxMeanId com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationLayerDataId_auxMeanId
#define auxStandardDeviationId com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationLayerDataId_auxStandardDeviationId
#define auxPopulationMeanId com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationLayerDataId_auxPopulationMeanId
#define auxPopulationVarianceId com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationLayerDataId_auxPopulationVarianceId

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers::batch_normalization;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationForwardResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_batch_1normalization_BatchNormalizationForwardResult_cNewResult
  (JNIEnv *env, jobject thisObj)
{
    return jniArgument<forward::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationForwardResult
 * Method:    cGetValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_batch_1normalization_BatchNormalizationForwardResult_cGetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if (id == auxDataId || id == auxWeightsId || id == auxMeanId || id == auxStandardDeviationId ||
        id == auxPopulationMeanId || id == auxPopulationVarianceId)
    {
        return jniArgument<forward::Result>::get<LayerDataId, Tensor>(resAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationForwardResult
 * Method:    cSetValue
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_batch_1normalization_BatchNormalizationForwardResult_cSetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    if (id == auxDataId || id == auxWeightsId || id == auxMeanId || id == auxStandardDeviationId ||
        id == auxPopulationMeanId || id == auxPopulationVarianceId)
    {
        jniArgument<forward::Result>::set<LayerDataId, Tensor>(resAddr, id, ntAddr);
    }
}
