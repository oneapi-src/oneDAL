/* file: result.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include "com_intel_daal_algorithms_optimization_solver_objective_function_Result.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_objective_function_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_objective_1function_Result_cNewResult(JNIEnv *, jobject)
{
    return jniArgument<objective_function::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_objective_function_Result
 * Method:    cGetResultNumericTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_objective_1function_Result_cGetResultNumericTable(JNIEnv *, jobject,
                                                                                                                              jlong resAddr, jint id)
{
    return jniArgument<objective_function::Result>::get<objective_function::ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_objective_function_Result
 * Method:    cSetResultNumericTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_objective_1function_Result_cSetResultNumericTable(JNIEnv *, jobject,
                                                                                                                             jlong resAddr, jint id,
                                                                                                                             jlong dcAddr)
{
    jniArgument<objective_function::Result>::set<objective_function::ResultId, NumericTable>(resAddr, id, dcAddr);
}
