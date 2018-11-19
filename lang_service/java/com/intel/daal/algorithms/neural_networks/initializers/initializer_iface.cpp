/* file: initializer_iface.cpp */
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
#include "neural_networks/initializers/JInitializerIface.h"

#include "daal.h"

using namespace daal;
using namespace daal::services;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_InitializerIface
 * Method:    cGetInput
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_InitializerIface_cGetInput
  (JNIEnv *env, jobject thisObj, jlong algAddr)
{
    return (jlong) & (staticPointerCast<initializers::InitializerIface, AlgorithmIface>(
        *((SharedPtr<AlgorithmIface> *)algAddr)))->input;
}
