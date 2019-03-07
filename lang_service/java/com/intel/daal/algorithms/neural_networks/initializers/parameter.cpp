/* file: parameter.cpp */
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
#include "neural_networks/initializers/JParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;


/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_Parameter
 * Method:    cSetLayer
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_Parameter_cSetLayer
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong layerAddr)
{
    (((initializers::Parameter *)cParameter))->layer = staticPointerCast<layers::forward::LayerIface, AlgorithmIface> (*(SharedPtr<AlgorithmIface> *)layerAddr);
}
