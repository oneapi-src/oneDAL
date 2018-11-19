/* file: distributed_step2_master_input.cpp */
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
#include "neural_networks/training/JDistributedStep2MasterInput.h"
#include "neural_networks/training/JDistributedStep2MasterInputId.h"

#include "daal.h"
#include "common_helpers.h"

#define partialResultsId com_intel_daal_algorithms_neural_networks_training_DistributedStep2MasterInputId_partialResultsId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep2MasterInput
 * Method:    cAddInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep2MasterInput_cAddInput
  (JNIEnv *, jobject, jlong inputAddr, jint id, jint key, jlong partialResultAddr)
{
    jniInput<training::DistributedInput<step2Master> >::add<training::Step2MasterInputId, training::PartialResult>(inputAddr, id, key, partialResultAddr);

}
