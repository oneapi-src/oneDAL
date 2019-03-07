/* file: eltwise_sum_backward_input.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
#include "neural_networks/layers/eltwise_sum/JEltwiseSumBackwardInput.h"

#include "daal.h"

#include "common_helpers.h"

using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms::neural_networks::layers::eltwise_sum;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumBackwardInput
 * Method:    cGetTensor
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumBackwardInput_cGetTensor
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<backward::Input>::get<LayerDataId, Tensor>(inputAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumBackwardInput
 * Method:    cSetTensor
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumBackwardInput_cSetTensor
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<backward::Input>::set<LayerDataId, Tensor>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumBackwardInput
 * Method:    cGetNumericTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumBackwardInput_cGetNumericTable
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<backward::Input>::get<LayerDataNumericTableId, NumericTable>(inputAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumBackwardInput
 * Method:    cSetNumericTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumBackwardInput_cSetNumericTable
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<backward::Input>::set<LayerDataNumericTableId, NumericTable>(inputAddr, id, ntAddr);
}
