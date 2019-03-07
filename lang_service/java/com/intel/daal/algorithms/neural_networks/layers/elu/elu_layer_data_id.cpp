/* file: elu_layer_data_id.cpp */
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
#include "neural_networks/layers/elu/JEluLayerDataId.h"

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
