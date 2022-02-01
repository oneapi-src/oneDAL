/* file: train_input.cpp */
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

#include "com_intel_daal_algorithms_linear_regression_training_Input.h"
#include "com_intel_daal_algorithms_linear_regression_training_DistributedStep2MasterInput.h"
#include "com_intel_daal_algorithms_linear_regression_training_TrainingMethod.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::linear_regression;

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_Input
 * Method:    cSetInput
 * Signature:(JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_Input_cSetInput(JNIEnv * env, jobject thisObj, jlong inputAddr,
                                                                                                  jint id, jlong ntAddr)
{
    jniInput<linear_regression::training::Input>::set<linear_regression::training::InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_Input
 * Method:    cGetInput
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_Input_cGetInput(JNIEnv * env, jobject thisObj, jlong inputAddr,
                                                                                                   jint id)
{
    return jniInput<linear_regression::training::Input>::get<linear_regression::training::InputId, NumericTable>(inputAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_DistributedStep2MasterInput
 * Method:    cAddInput
 * Signature:(JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_DistributedStep2MasterInput_cAddInput(JNIEnv * env, jobject thisObj,
                                                                                                                        jlong inputAddr, jint id,
                                                                                                                        jlong presAddr)
{
    jniInput<linear_regression::training::DistributedInput<step2Master> >::add<linear_regression::training::Step2MasterInputId,
                                                                               training::PartialResult>(inputAddr, id, presAddr);
}
