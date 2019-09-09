/* file: elu_layer_data_id.cpp */
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
#include "com_intel_daal_algorithms_neural_networks_layers_elu_EluLayerDataId.h"

#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_elu_EluLayerDataId
 * Method:    cGetAuxDataId
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_elu_EluLayerDataId_cGetAuxDataId
  (JNIEnv *, jclass)
{
    return (jint)elu::auxData;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_elu_EluLayerDataId
 * Method:    cGetAuxIntermediateValueId
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_elu_EluLayerDataId_cGetAuxIntermediateValueId
  (JNIEnv *, jclass)
{
    return (jint)elu::auxIntermediateValue;
}
