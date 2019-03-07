/* file: parameter.cpp */
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

#include "daal.h"

#include "optimization_solver/objective_function/JParameter.h"

#include "common_defines.i"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_objective_function_Parameter
 * Method:    cSetResultsToCompute
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_objective_1function_Parameter_cSetResultsToCompute
(JNIEnv *, jobject, jlong parAddr, jlong resultsToCompute)
{
    ((objective_function::Parameter *)parAddr)->resultsToCompute = resultsToCompute;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_objective_function_Parameter
 * Method:    cGetResultsToCompute
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_objective_1function_Parameter_cGetResultsToCompute
(JNIEnv *, jobject, jlong parAddr)
{
    return ((objective_function::Parameter *)parAddr)->resultsToCompute;
}
