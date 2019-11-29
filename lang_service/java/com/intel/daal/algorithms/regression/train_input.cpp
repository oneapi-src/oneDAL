/* file: train_input.cpp */
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
#include "com_intel_daal_algorithms_ComputeMode.h"
#include "com_intel_daal_algorithms_regression_training_TrainingInput.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::regression;
using namespace daal::algorithms::regression::training;

#define jBatch  com_intel_daal_algorithms_ComputeMode_batchValue
#define jOnline com_intel_daal_algorithms_ComputeMode_onlineValue

/*
 * Class:     com_intel_daal_algorithms_regression_training_TrainingInput
 * Method:    cInit
 * Signature: (JIJ)I
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_regression_training_TrainingInput_cInit(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                               jint cmode)
{
    regression::training::Input * inputPtr = NULL;

    if (cmode == jBatch)
    {
        SharedPtr<Batch> alg = staticPointerCast<Batch, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
        inputPtr             = alg->getInput();
    }
    else if (cmode == jOnline)
    {
        SharedPtr<Online> alg = staticPointerCast<Online, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
        inputPtr              = alg->getInput();
    }

    return (jlong)inputPtr;
}

/*
 * Class:     Java_com_intel_daal_algorithms_regression_training_TrainingInput
 * Method:    cSetInput
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_regression_training_TrainingInput_cSetInput(JNIEnv * env, jobject thisObj, jlong inputAddr,
                                                                                                  jint id, jlong ntAddr)
{
    jniInput<regression::training::Input>::set<regression::training::InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_regression_training_TrainingInput
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_regression_training_TrainingInput_cGetInputTable(JNIEnv * env, jobject thisObj,
                                                                                                        jlong inputAddr, jint id)
{
    return jniInput<regression::training::Input>::get<regression::training::InputId, NumericTable>(inputAddr, id);
}
