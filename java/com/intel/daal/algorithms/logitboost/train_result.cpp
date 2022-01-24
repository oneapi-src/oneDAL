/* file: train_result.cpp */
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

#include "daal.h"
#include "com/intel/daal/common_helpers.h"
#include "com_intel_daal_algorithms_classifier_training_TrainingResultId.h"
#include "com_intel_daal_algorithms_logitboost_training_TrainingResult.h"

using namespace daal;
using namespace daal::algorithms;

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logitboost_training_TrainingResult_cGetModel(JNIEnv *, jobject, jlong resAddr, jint id)
{
    return jniArgument<logitboost::training::Result>::get<classifier::training::ResultId, logitboost::Model>(resAddr, classifier::training::model);
}
