/* file: eltwise_sum_layer_data_numeric_table_id.cpp */
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
#include "neural_networks/layers/eltwise_sum/JEltwiseSumLayerDataNumericTableId.h"

#include "daal.h"
#include "common_helpers.h"

using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumLayerDataNumericTableId
 * Method:    cGetAuxNumberOfCoefficientsId
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumLayerDataNumericTableId_cGetAuxNumberOfCoefficientsId
  (JNIEnv *env, jclass)
{
    return (jint)eltwise_sum::auxNumberOfCoefficients;
}
