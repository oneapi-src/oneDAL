/* file: batch.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#include "optimization_solver/iterative_solver/JBatch.h"

#include "common_defines.i"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_iterative_1solver_Batch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Batch_cClone
(JNIEnv *, jobject, jlong algAddr)
{
    services::SharedPtr<AlgorithmIface> *ptr = new services::SharedPtr<AlgorithmIface>();
    *ptr = staticPointerCast<iterative_solver::Batch, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->clone();
    return (jlong)ptr;

}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_iterative_1solver_Batch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Batch_cGetInput
(JNIEnv *, jobject, jlong algAddr)
{
    return (jlong) & staticPointerCast<iterative_solver::Batch, AlgorithmIface > (*(SharedPtr<AlgorithmIface> *)algAddr)->input;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_iterative_1solver_Batch
 * Method:    cGetParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Batch_cGetParameter
(JNIEnv *, jobject, jlong algAddr)
{
    return (jlong) & (staticPointerCast<iterative_solver::Batch, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr))->parameter;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_iterative_1solver_Batch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Batch_cGetResult
(JNIEnv *, jobject, jlong algAddr)
{
    SerializationIfacePtr *ptr = new SerializationIfacePtr();
    *ptr = staticPointerCast<iterative_solver::Batch, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr)->getResult();

    return (jlong)ptr;
}
