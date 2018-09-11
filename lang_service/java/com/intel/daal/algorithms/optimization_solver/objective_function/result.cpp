/* file: result.cpp */
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
 * Method:    cGetResultNumericTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_objective_1function_Result_cGetResultNumericTable
(JNIEnv *, jobject, jlong resAddr, jint id)
{
    return jniArgument<objective_function::Result>::get<objective_function::ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_objective_function_Result
 * Method:    cSetResultNumericTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_objective_1function_Result_cSetResultNumericTable
(JNIEnv *, jobject, jlong resAddr, jint id, jlong dcAddr)
{
    jniArgument<objective_function::Result>::set<objective_function::ResultId, NumericTable>(resAddr, id, dcAddr);
}
