/* file: result.cpp */
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

#include "optimization_solver/objective_function/JResult.h"

#include "common_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_objective_function_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_objective_1function_Result_cNewResult
(JNIEnv *, jobject)
{
    return jniArgument<objective_function::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_objective_function_Result
 * Method:    cGetResultDataCollection
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_objective_1function_Result_cGetResultDataCollection
(JNIEnv *, jobject, jlong resAddr, jint id)
{
    return jniArgument<objective_function::Result>::get<objective_function::ResultId, DataCollection>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_objective_function_Result
 * Method:    cSetResultDataCollection
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_objective_1function_Result_cSetResultDataCollection
(JNIEnv *, jobject, jlong resAddr, jint id, jlong dcAddr)
{
    jniArgument<objective_function::Result>::set<objective_function::ResultId, DataCollection>(resAddr, id, dcAddr);
}


/*
 * Class:     com_intel_daal_algorithms_optimization_solver_objective_function_Result
 * Method:    cGetResultTable
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_objective_1function_Result_cGetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jint idx)
{
    NumericTablePtr *nt = new NumericTablePtr();
    SerializationIfacePtr *res = (SerializationIfacePtr *)resAddr;
    objective_function::Result *result = (objective_function::Result *)(res->get());
    *nt = result->get(objective_function::resultCollection, (objective_function::ResultCollectionId)idx);
    return (jlong)nt;
}
