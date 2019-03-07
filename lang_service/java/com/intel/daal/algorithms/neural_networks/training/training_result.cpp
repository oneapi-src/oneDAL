/* file: training_result.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
#include "neural_networks/training/JTrainingResult.h"
#include "neural_networks/training/JTrainingResultId.h"

#include "daal.h"

#include "common_helpers.h"

#define modelId com_intel_daal_algorithms_neural_networks_training_TrainingResultId_modelId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingResult_cNewResult
  (JNIEnv *env, jobject thisObj)
{
    return jniArgument<training::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingResult
 * Method:    cGetValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingResult_cGetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if (id == modelId)
    {
        return jniArgument<training::Result>::get<training::ResultId, training::Model>(resAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingResult
 * Method:    cSetValue
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingResult_cSetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong modelAddr)
{
    if (id == modelId)
    {
        jniArgument<training::Result>::set<training::ResultId, training::Model>(resAddr, id, modelAddr);
    }
}
