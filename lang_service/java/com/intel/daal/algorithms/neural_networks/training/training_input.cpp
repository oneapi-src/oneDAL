/* file: training_input.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
