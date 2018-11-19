/* file: batch.cpp */
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
    return (jlong)(staticPointerCast<iterative_solver::Batch, AlgorithmIface > (*(SharedPtr<AlgorithmIface> *)algAddr))->getInput();
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_iterative_1solver_Batch
 * Method:    cGetParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Batch_cGetParameter
(JNIEnv *, jobject, jlong algAddr)
{
    return (jlong)(staticPointerCast<iterative_solver::Batch, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr))->getParameter();
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
