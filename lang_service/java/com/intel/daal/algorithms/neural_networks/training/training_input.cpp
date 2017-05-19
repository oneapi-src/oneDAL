/* file: training_input.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
#include "neural_networks/training/JTrainingInput.h"
#include "neural_networks/training/JTrainingInputId.h"
#include "neural_networks/training/JTrainingInputCollectionId.h"

#include "daal.h"

#include "common_helpers.h"

#define dataId com_intel_daal_algorithms_neural_networks_training_TrainingInputId_dataId
#define groundTruthId com_intel_daal_algorithms_neural_networks_training_TrainingInputId_groundTruthId
#define groundTruthCollectionId com_intel_daal_algorithms_neural_networks_training_TrainingInputCollectionId_groundTruthCollectionId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingInput_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if (id == dataId || id == groundTruthId)
    {
        jniInput<training::Input>::set<training::InputId, Tensor>(inputAddr, id, ntAddr);
    }
    else if (id == groundTruthCollectionId)
    {
        jniInput<training::Input>::set<training::InputCollectionId, KeyValueDataCollection>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingInput
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingInput_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if (id == dataId || id == groundTruthId)
    {
        return jniInput<training::Input>::get<training::InputId, Tensor>(inputAddr, id);
    }
    else if (id == groundTruthCollectionId)
    {
        return jniInput<training::Input>::get<training::InputCollectionId, KeyValueDataCollection>(inputAddr, id);
    }

    return (jlong)0;
}


/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingInput
 * Method:    cAddTensor
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingInput_cAddTensor
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jint key, jlong ntAddr)
{
    if (id == groundTruthCollectionId)
    {
        SerializationIfacePtr *collectionShPtr = (SerializationIfacePtr *)(jniInput<training::Input>::get<training::InputCollectionId, KeyValueDataCollection>(inputAddr, id));
        SerializationIfacePtr *valueShPtr = (SerializationIfacePtr *)ntAddr;
        KeyValueDataCollection *collection = static_cast<KeyValueDataCollection *>(collectionShPtr->get());
        (*collection)[(size_t)key] = *valueShPtr;
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingInput
 * Method:    cGetTensor
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingInput_cGetTensor
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jint key)
{
    if (id == groundTruthCollectionId)
    {
        SerializationIfacePtr *collectionShPtr = (SerializationIfacePtr *)(jniInput<training::Input>::get<training::InputCollectionId, KeyValueDataCollection>(inputAddr, id));
        KeyValueDataCollection *collection = static_cast<KeyValueDataCollection *>(collectionShPtr->get());
        SerializationIfacePtr *value = new SerializationIfacePtr((*collection)[(size_t)key]);
        return (jlong)value;
    }

    return (jlong)0;
}
