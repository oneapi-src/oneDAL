/* file: input.cpp */
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

#include "optimization_solver/sum_of_functions/JInput.h"

#include "common_defines.i"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sum_of_functions_Input
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sum_1of_1functions_Input_cSetInput
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<sum_of_functions::Input>::set<sum_of_functions::InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sum_of_functions_Input
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sum_1of_1functions_Input_cGetInput
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<sum_of_functions::Input>::get<sum_of_functions::InputId, NumericTable>(inputAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sum_of_functions_Input
 * Method:    cSetCInput
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sum_1of_1functions_Input_cSetCInput
(JNIEnv *, jobject, jlong inputAddr, jlong algAddr)
{
    sum_of_functions::Input *inputPtr = (sum_of_functions::Input *)inputAddr;

    SharedPtr<sum_of_functions::Batch> alg =
        staticPointerCast<sum_of_functions::Batch, AlgorithmIface>
        (*(SharedPtr<AlgorithmIface> *)algAddr);
    alg->sumOfFunctionsInput = inputPtr;
}


/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sum_of_functions_Input
 * Method:    cCreateInput
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sum_1of_1functions_Input_cCreateInput
(JNIEnv *, jobject)
{
    jlong addr = 0;
    addr = (jlong)(new sum_of_functions::Input());
    return addr;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sum_of_functions_Input
 * Method:    cInputDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sum_1of_1functions_Input_cInputDispose
(JNIEnv *, jobject, jlong createdInput)
{
    sum_of_functions::Input* ptr = (sum_of_functions::Input *) createdInput;
    delete ptr;
}
