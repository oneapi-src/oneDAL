/* file: result.cpp */
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

#include "optimization_solver/iterative_solver/JResult.h"

#include "common_defines.i"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_iterative_solver_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Result_cNewResult
(JNIEnv *, jobject)
{
    return jniArgument<optimization_solver::iterative_solver::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_iterative_solver_Result
 * Method:    cGetResultTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Result_cGetResultTable
(JNIEnv *, jobject, jlong resAddr, jint id)
{
    return jniArgument<optimization_solver::iterative_solver::Result>::get<
        optimization_solver::iterative_solver::ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_iterative_solver_Result
 * Method:    cSetResultTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Result_cSetResultTable
(JNIEnv *, jobject, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<optimization_solver::iterative_solver::Result>::set<
        optimization_solver::iterative_solver::ResultId, NumericTable>(resAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_iterative_solver_Result
* Method:    cGetOptionalResult
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Result_cGetOptionalResult
(JNIEnv *, jobject, jlong resAddr, jint id)
{
    return jniArgument<optimization_solver::iterative_solver::Result>::get<
        optimization_solver::iterative_solver::OptionalResultId, OptionalArgument>(resAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_iterative_solver_Result
* Method:    cSetOptionalResult
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Result_cSetOptionalResult
(JNIEnv *, jobject, jlong resAddr, jint id, jlong argAddr)
{
    jniArgument<optimization_solver::iterative_solver::Result>::set<
        optimization_solver::iterative_solver::OptionalResultId, OptionalArgument>(resAddr, id, argAddr);
}
