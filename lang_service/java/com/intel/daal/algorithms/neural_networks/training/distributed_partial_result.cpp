/* file: distributed_partial_result.cpp */
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
#include "neural_networks/training/JDistributedPartialResult.h"
#include "neural_networks/training/JDistributedPartialResultId.h"

#include "daal.h"
#include "common_helpers.h"

#define resultFromMasterId com_intel_daal_algorithms_neural_networks_training_DistributedPartialResultId_resultFromMasterId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedPartialResult
 * Method:    cNewPartialResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedPartialResult_cNewPartialResult
  (JNIEnv *, jobject)
{
    return jniArgument<training::DistributedPartialResult>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedPartialResult
 * Method:    cGetResult
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedPartialResult_cGetResult
  (JNIEnv *, jobject, jlong algAddr, jint id)
{
    if (id == resultFromMasterId)
    {
        return jniArgument<training::DistributedPartialResult>::get<training::Step2MasterPartialResultId, training::Result>(algAddr, id);
    }

    return (jlong)0;
}
