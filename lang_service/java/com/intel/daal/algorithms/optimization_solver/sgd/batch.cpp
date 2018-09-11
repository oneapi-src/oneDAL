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

#include "optimization_solver/sgd/JBatch.h"

#include "common_defines.i"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_Batch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_Batch_cInit
(JNIEnv *, jobject, jint prec, jint method)
{
    return jniBatch<sgd::Method, sgd::Batch, sgd::defaultDense, sgd::miniBatch, sgd::momentum>::newObj(prec, method, SharedPtr<sum_of_functions::Batch>());
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_Batch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_Batch_cClone
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<sgd::Method, sgd::Batch, sgd::defaultDense, sgd::miniBatch, sgd::momentum>::getClone(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_Batch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_Batch_cGetInput
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<sgd::Method, sgd::Batch, sgd::defaultDense, sgd::miniBatch, sgd::momentum>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_Batch
 * Method:    cGetParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_Batch_cGetParameter
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<sgd::Method, sgd::Batch, sgd::defaultDense, sgd::miniBatch, sgd::momentum>::getParameter(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_sgd_Result
* Method:    cNewResult
* Signature: ()J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_Result_cNewResult
(JNIEnv *, jobject)
{
    return jniArgument<sgd::Result>::newObj();
}
