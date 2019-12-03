/* file: input.cpp */
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

#include "daal.h"

#include "com_intel_daal_algorithms_optimization_solver_logistic_loss_Input.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_logistic_loss_Input
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_logistic_1loss_Input_cSetInput(JNIEnv *, jobject, jlong inputAddr, jint id,
                                                                                                          jlong ntAddr)
{
    jniInput<logistic_loss::Input>::set<logistic_loss::InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_logistic_loss_Input
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_logistic_1loss_Input_cGetInput(JNIEnv *, jobject, jlong inputAddr,
                                                                                                           jint id)
{
    return jniInput<logistic_loss::Input>::get<logistic_loss::InputId, NumericTable>(inputAddr, id);
}
