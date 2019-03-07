/* file: lcn_forward_result.cpp */
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
#include "neural_networks/layers/lcn/JLcnForwardResult.h"
#include "neural_networks/layers/lcn/JLcnLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define auxCenteredDataId com_intel_daal_algorithms_neural_networks_layers_lcn_LcnLayerDataId_auxCenteredDataId
#define auxSigmaId        com_intel_daal_algorithms_neural_networks_layers_lcn_LcnLayerDataId_auxSigmaId
#define auxCId            com_intel_daal_algorithms_neural_networks_layers_lcn_LcnLayerDataId_auxCId
#define auxInvMaxId       com_intel_daal_algorithms_neural_networks_layers_lcn_LcnLayerDataId_auxInvMaxId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers::lcn;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lcn_LcnForwardResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lcn_LcnForwardResult_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<forward::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lcn_LcnForwardResult
 * Method:    cGetValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lcn_LcnForwardResult_cGetValue
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if (id == auxCenteredDataId || id == auxSigmaId || id == auxCId || id == auxInvMaxId)
    {
        return jniArgument<forward::Result>::get<LayerDataId, Tensor>(resAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lcn_LcnForwardResult
 * Method:    cSetValue
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lcn_LcnForwardResult_cSetValue
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    if (id == auxCenteredDataId || id == auxSigmaId || id == auxCId || id == auxInvMaxId)
    {
        jniArgument<forward::Result>::set<LayerDataId, Tensor>(resAddr, id, ntAddr);
    }
}
