/* file: train_batch.cpp */
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

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>/* Header for class com_intel_daal_algorithms_weak_learner_training_TrainingResult */

#include "daal.h"
#include "com_intel_daal_algorithms_weak_learner_training_TrainingBatch.h"

using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms;
using namespace daal::services;

#include "train_types.i"

/*
 * Class:     com_intel_daal_algorithms_weak_learner_training_TrainingBatch
 * Method:    cGetResult
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_weak_1learner_training_TrainingBatch_cGetResult(JNIEnv * env, jobject thisObj, jlong algAddr)
{
    SerializationIfacePtr * ptr = new SerializationIfacePtr();

    SharedPtr<wl_tr_of> alg = staticPointerCast<wl_tr_of, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
    *ptr                    = alg->getResult();

    return (jlong)ptr;
}
