/* file: distributed_step1_local_input.cpp */
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
#include "neural_networks/training/JDistributedStep1LocalInput.h"
#include "neural_networks/training/JDistributedStep1LocalInputId.h"

#include "daal.h"
#include "common_helpers.h"

#define dataId com_intel_daal_algorithms_neural_networks_training_TrainingInputId_dataId
#define groundTruthId com_intel_daal_algorithms_neural_networks_training_TrainingInputId_groundTruthId
#define inputModelId com_intel_daal_algorithms_neural_networks_training_DistributedStep1LocalInputId_inputModelId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep1LocalInput
 * Method:    cSetModel
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep1LocalInput_cSetModel
  (JNIEnv *, jobject, jlong inputAddr, jint id, jlong modelAddr)
{
    if (id == inputModelId)
    {
        jniInput<training::DistributedInput<step1Local> >::set<training::Step1LocalInputId, training::Model>(inputAddr, id, modelAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep1LocalInput
 * Method:    cGetModel
 * Signature: (JIJ)V
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep1LocalInput_cGetModel
  (JNIEnv *, jobject, jlong inputAddr, jint id)
{
    if (id == inputModelId)
    {
        return jniInput<training::DistributedInput<step1Local> >::get<training::Step1LocalInputId, training::Model>(inputAddr, id);
    }
    return (jlong)0;
}
