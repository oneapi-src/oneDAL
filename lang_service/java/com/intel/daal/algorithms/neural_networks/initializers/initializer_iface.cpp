/* file: initializer_iface.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
