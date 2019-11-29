/* file: batch_base.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#include "com_intel_daal_algorithms_distributions_BatchBase.h"

#include "daal.h"

using namespace daal;
using namespace daal::services;
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_distributions_BatchBase
 * Method:    cGetInput
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_distributions_BatchBase_cGetInput(JNIEnv * env, jobject thisObj, jlong algAddr)
{
    return (jlong) & (staticPointerCast<distributions::BatchBase, AlgorithmIface>(*((SharedPtr<AlgorithmIface> *)algAddr)))->input;
}
