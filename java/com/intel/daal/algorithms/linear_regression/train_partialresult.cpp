/* file: train_partialresult.cpp */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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

#include "com_intel_daal_algorithms_linear_regression_training_PartialResult.h"
#include "com_intel_daal_algorithms_linear_regression_training_TrainingMethod.h"

#include "com/intel/daal/common_helpers.h"

#include "com_intel_daal_algorithms_linear_regression_training_PartialResultId.h"
#define ModelId com_intel_daal_algorithms_linear_regression_training_PartialResultId_ModelId

USING_COMMON_NAMESPACES();

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_PartialResult
 * Method:    cNewPartialResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_PartialResult_cNewPartialResult(JNIEnv * env, jobject thisObj)
{
    return jniArgument<linear_regression::training::PartialResult>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_PartialResult
 * Method:    cGetModel
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_PartialResult_cGetModel(JNIEnv * env, jobject thisObj,
                                                                                                           jlong resAddr, jint id)
{
    if (id == ModelId)
    {
        return jniArgument<linear_regression::training::PartialResult>::get<linear_regression::training::PartialResultID, linear_regression::Model>(
            resAddr, id);
    }

    return (jlong)0;
}
