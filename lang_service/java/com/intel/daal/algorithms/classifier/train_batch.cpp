/* file: train_batch.cpp */
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
#include "com_intel_daal_algorithms_classifier_training_TrainingBatch.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_classifier_training_TrainingBatch_cGetResult(JNIEnv *, jobject, jlong self)
{
    SerializationIfacePtr * ptr                = new SerializationIfacePtr();
    SharedPtr<classifier::training::Batch> alg = staticPointerCast<classifier::training::Batch>(*(SharedPtr<AlgorithmIface> *)self);
    *ptr                                       = alg->getResult();
    return (jlong)ptr;
}
