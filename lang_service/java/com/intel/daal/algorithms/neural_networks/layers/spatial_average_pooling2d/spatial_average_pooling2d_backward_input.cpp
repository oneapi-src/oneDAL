/* file: spatial_average_pooling2d_backward_input.cpp */
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
#include "neural_networks/layers/spatial_average_pooling2d/JSpatialAveragePooling2dBackwardInput.h"
#include "neural_networks/layers/spatial_average_pooling2d/JSpatialAveragePooling2dLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define auxInputDimensionsId com_intel_daal_algorithms_neural_networks_layers_spatial_average_pooling2d_SpatialAveragePooling2dLayerDataId_auxInputDimensionsId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers::spatial_average_pooling2d;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_average_1pooling2d_SpatialAveragePooling2dBackwardInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1average_1pooling2d_SpatialAveragePooling2dBackwardInput_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if (id == auxInputDimensionsId)
    {
        jniInput<backward::Input>::set<LayerDataId, NumericTable>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_average_1pooling2d_SpatialAveragePooling2dBackwardInput
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1average_1pooling2d_SpatialAveragePooling2dBackwardInput_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if (id == auxInputDimensionsId)
    {
        return jniInput<backward::Input>::get<LayerDataId, NumericTable>(inputAddr, id);
    }

    return (jlong)0;
}
