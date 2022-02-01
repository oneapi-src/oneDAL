/* file: onlineparameter.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "daal.h"
#include "com_intel_daal_algorithms_covariance_OnlineParameter.h"

#include "covariance_types.i"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::covariance;

/*
 * Class:     com_intel_daal_algorithms_covariance_Parameter
 * Method:    cSetCParameterObject
 * Signature: (JJIIII)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_covariance_OnlineParameter_cSetCParameterObject(JNIEnv * env, jobject thisObj,
                                                                                                      jlong parameterAddr, jlong algAddr)
{
    using namespace daal::services;
    using namespace daal::algorithms;
    using namespace daal::data_management;
    staticPointerCast<covariance::OnlineImpl, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->parameter =
        *((covariance::OnlineParameter *)parameterAddr);
}
