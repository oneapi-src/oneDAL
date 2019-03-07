/* file: maximum_pooling1d_backward_result.cpp */
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
#include "neural_networks/layers/maximum_pooling1d/JMaximumPooling1dBackwardResult.h"
#include "neural_networks/layers/maximum_pooling1d/JMaximumPooling1dLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

using namespace daal;
using namespace daal::algorithms::neural_networks::layers::maximum_pooling1d;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_maximum_1pooling1d_MaximumPooling1dBackwardResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_maximum_1pooling1d_MaximumPooling1dBackwardResult_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<backward::Result>::newObj();
}
