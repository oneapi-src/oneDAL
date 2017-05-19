/* file: partial_result.cpp */
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
