/* file: partial_result.cpp */
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
#include "neural_networks/training/JPartialResult.h"
#include "neural_networks/training/JPartialResultId.h"

#include "daal.h"

#include "common_helpers.h"

#define derivativesId com_intel_daal_algorithms_neural_networks_training_PartialResultId_derivativesId
#define batchSizeId com_intel_daal_algorithms_neural_networks_training_PartialResultId_batchSizeId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_PartialResult
 * Method:    cNewPartialResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_PartialResult_cNewPartialResult
  (JNIEnv *, jobject)
{
    return jniArgument<training::PartialResult>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_PartialResult
 * Method:    cGetPartialResultTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_PartialResult_cGetPartialResultTable
  (JNIEnv *, jobject, jlong resAddr, jint id)
{
    if(id == derivativesId || id == batchSizeId)
    {
        return jniArgument<training::PartialResult>::get<training::Step1LocalPartialResultId, NumericTable>(resAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_PartialResult
 * Method:    cSetPartialResultTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_PartialResult_cSetPartialResultTable
  (JNIEnv *, jobject, jlong resAddr, jint id, jlong ntAddr)
{
    if(id == derivativesId || id == batchSizeId)
    {
        jniArgument<training::PartialResult>::set<training::Step1LocalPartialResultId, NumericTable>(resAddr, id, ntAddr);
    }
}
