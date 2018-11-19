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

#include "optimization_solver/cross_entropy_loss/JBatch.h"

#include "common_defines.i"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_cross_entropy_loss_Batch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_entropy_loss_Batch_cInit
(JNIEnv *, jobject, jint prec, jint method, jlong numberOfTerms, jlong nClasses)
{
    return jniBatch<cross_entropy_loss::Method, cross_entropy_loss::Batch, cross_entropy_loss::defaultDense>::newObj(prec, method, numberOfTerms, nClasses);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_cross_entropy_loss_Batch
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_entropy_loss_Batch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<cross_entropy_loss::Method, cross_entropy_loss::Batch, cross_entropy_loss::defaultDense>::getClone(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_cross_entropy_loss_Batch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_entropy_loss_Batch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<cross_entropy_loss::Method, cross_entropy_loss::Batch, cross_entropy_loss::defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_cross_entropy_loss_Batch
 * Method:    cGetParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_entropy_loss_Batch_cGetParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<cross_entropy_loss::Method, cross_entropy_loss::Batch, cross_entropy_loss::defaultDense>::getParameter(prec, method, algAddr);
}
