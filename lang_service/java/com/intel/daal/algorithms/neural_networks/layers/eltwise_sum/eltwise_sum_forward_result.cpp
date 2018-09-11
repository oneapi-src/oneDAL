/* file: eltwise_sum_forward_result.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#include <jni.h>
#include "neural_networks/layers/eltwise_sum/JEltwiseSumForwardResult.h"
#include "neural_networks/layers/eltwise_sum/JEltwiseSumLayerDataId.h"

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
