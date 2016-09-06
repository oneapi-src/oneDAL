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

#include "optimization_solver/objective_function/JBatch.h"

#include "common_defines.i"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_objective_function_Batch
 * Method:    cSetResult
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_objective_1function_Batch_cSetResult
(JNIEnv *, jobject, jlong algAddr, jlong cResult)
{
    SharedPtr<objective_function::Batch> alg =
        staticPointerCast<objective_function::Batch, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);

    SerializationIfacePtr *serializableShPtr = (SerializationIfacePtr *)cResult;
    SharedPtr<objective_function::Result> resultShPtr =
        staticPointerCast<objective_function::Result, SerializationIface>(*serializableShPtr);

    alg->setResult(resultShPtr);
}



/*
 * Class:     com_intel_daal_algorithms_optimization_solver_objective_function_Batch
 * Method:    cGetResult
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_objective_1function_Batch_cGetResult
(JNIEnv *, jobject, jlong algAddr)
{
    SharedPtr<objective_function::Batch> alg =
        staticPointerCast<objective_function::Batch, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);

    SerializationIfacePtr *resShPtr = new SerializationIfacePtr();
    *resShPtr = staticPointerCast<SerializationIface, objective_function::Result>(alg->getResult());
    return (jlong)(resShPtr);
}
