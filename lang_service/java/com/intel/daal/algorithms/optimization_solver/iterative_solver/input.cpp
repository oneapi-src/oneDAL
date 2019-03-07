/* file: input.cpp */
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

#include "optimization_solver/iterative_solver/JInput.h"

#include "common_defines.i"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_iterative_solver_Input
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Input_cSetInput
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<iterative_solver::Input>::set<iterative_solver::InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_iterative_solver_Input
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Input_cGetInput
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<iterative_solver::Input>::get<iterative_solver::InputId, NumericTable>(inputAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_iterative_solver_Input
* Method:    cSetOptionalInput
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Input_cSetOptionalInput
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong argAddr)
{
    jniInput<iterative_solver::Input>::set<iterative_solver::OptionalInputId, OptionalArgument>(inputAddr, id, argAddr);
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_iterative_solver_Input
* Method:    cGetOptionalInput
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Input_cGetOptionalInput
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<iterative_solver::Input>::get<iterative_solver::OptionalInputId, OptionalArgument>(inputAddr, id);
}
