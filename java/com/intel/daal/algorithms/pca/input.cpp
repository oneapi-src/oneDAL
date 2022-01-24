/* file: input.cpp */
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

#include <jni.h>
#include "daal.h"

#include "com_intel_daal_algorithms_pca_Input.h"
#include "com_intel_daal_algorithms_pca_DistributedStep2MasterInput.h"
#include "com_intel_daal_algorithms_pca_Method.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::pca;

/*
 * Class:     com_intel_daal_algorithms_pca_Input
 * Method:    cSetInputTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_Input_cSetInputTable(JNIEnv * env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<pca::Input>::set<pca::InputDatasetId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_Input
 * Method:    cSetInputCorrelation
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_Input_cSetInputCorrelation(JNIEnv * env, jobject thisObj, jlong inputAddr, jint id,
                                                                                     jlong ntAddr)
{
    jniInput<pca::Input>::set<pca::InputCorrelationId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_pca_Input
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_Input_cGetInputTable(JNIEnv * env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<pca::Input>::get<pca::InputDatasetId, NumericTable>(inputAddr, id);
}
