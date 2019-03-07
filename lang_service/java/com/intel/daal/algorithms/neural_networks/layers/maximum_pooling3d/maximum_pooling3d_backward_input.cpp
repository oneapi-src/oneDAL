/* file: maximum_pooling3d_backward_input.cpp */
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
#include "neural_networks/layers/maximum_pooling3d/JMaximumPooling3dBackwardInput.h"
#include "neural_networks/layers/maximum_pooling3d/JMaximumPooling3dLayerDataId.h"
#include "neural_networks/layers/maximum_pooling3d/JMaximumPooling3dLayerDataNumericTableId.h"

#include "daal.h"

#include "common_helpers.h"

#define auxSelectedIndicesId com_intel_daal_algorithms_neural_networks_layers_maximum_pooling3d_MaximumPooling3dLayerDataId_auxSelectedIndicesId
#define auxInputDimensionsId com_intel_daal_algorithms_neural_networks_layers_maximum_pooling3d_MaximumPooling3dLayerDataNumericTableId_auxInputDimensionsId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers::maximum_pooling3d;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_maximum_1pooling3d_MaximumPooling3dBackwardInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_maximum_1pooling3d_MaximumPooling3dBackwardInput_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if (id == auxSelectedIndicesId)
    {
        jniInput<backward::Input>::set<LayerDataId, Tensor>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_maximum_1pooling3d_MaximumPooling3dBackwardInput
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_maximum_1pooling3d_MaximumPooling3dBackwardInput_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if (id == auxSelectedIndicesId)
    {
        return jniInput<backward::Input>::get<LayerDataId, Tensor>(inputAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_maximum_pooling3d_MaximumPooling3dBackwardInput
 * Method:    cSetInputNumericTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_maximum_1pooling3d_MaximumPooling3dBackwardInput_cSetInputNumericTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if (id == auxInputDimensionsId)
    {
        jniInput<backward::Input>::set<LayerDataNumericTableId, NumericTable>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_maximum_pooling3d_MaximumPooling3dBackwardInput
 * Method:    cGetInputNumericTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_maximum_1pooling3d_MaximumPooling3dBackwardInput_cGetInputNumericTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if (id == auxInputDimensionsId)
    {
        return jniInput<backward::Input>::get<LayerDataNumericTableId, NumericTable>(inputAddr, id);
    }

    return (jlong)0;
}
