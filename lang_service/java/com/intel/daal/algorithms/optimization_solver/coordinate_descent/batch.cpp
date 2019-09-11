/* file: batch.cpp */
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

#include "com_intel_daal_algorithms_optimization_solver_coordinate_descent_Batch.h"
#include "com_intel_daal_algorithms_optimization_solver_coordinate_descent_Result.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Batch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Batch_cInit
(JNIEnv *, jobject, jint prec, jint method)
{
    return jniBatch<coordinate_descent::Method, coordinate_descent::Batch, coordinate_descent::defaultDense>::newObj(prec, method, SharedPtr<sum_of_functions::Batch>());
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Batch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Batch_cClone
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<coordinate_descent::Method, coordinate_descent::Batch, coordinate_descent::defaultDense>::getClone(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Batch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Batch_cGetInput
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<coordinate_descent::Method, coordinate_descent::Batch, coordinate_descent::defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Batch
 * Method:    cGetParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Batch_cGetParameter
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<coordinate_descent::Method, coordinate_descent::Batch, coordinate_descent::defaultDense>::getParameter(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Result
* Method:    cNewResult
* Signature: ()J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Result_cNewResult
(JNIEnv *, jobject)
{
    return jniArgument<coordinate_descent::Result>::newObj();
}
