/* file: eltwise_sum_forward_result.cpp */
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
#include "com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumForwardResult.h"

#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers::eltwise_sum;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumForwardResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumForwardResult_cNewResult
  (JNIEnv *env, jobject thisObj)
{
    return jniArgument<forward::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_EltwiseSumForwardResult
 * Method:    cGetTensor
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumForwardResult_cGetTensor
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<forward::Result>::get<LayerDataId, Tensor>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_EltwiseSumForwardResult
 * Method:    cSetTensor
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumForwardResult_cSetTensor
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong tensorAddr)
{
    jniArgument<forward::Result>::set<LayerDataId, Tensor>(resAddr, id, tensorAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_EltwiseSumForwardResult
 * Method:    cGetNumericTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumForwardResult_cGetNumericTable
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<forward::Result>::get<LayerDataNumericTableId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_EltwiseSumForwardResult
 * Method:    cSetNumericTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumForwardResult_cSetNumericTable
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<forward::Result>::set<LayerDataNumericTableId, NumericTable>(resAddr, id, ntAddr);
}
