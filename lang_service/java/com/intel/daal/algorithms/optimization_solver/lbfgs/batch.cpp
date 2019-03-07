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

#include "optimization_solver/lbfgs/JBatch.h"
#include "optimization_solver/lbfgs/JInput.h"
#include "optimization_solver/lbfgs/JResult.h"

#include "common_defines.i"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Batch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Batch_cInit
  (JNIEnv *, jobject, jint prec, jint method)
{
    return jniBatch<lbfgs::Method, lbfgs::Batch, lbfgs::defaultDense>::newObj(prec, method, SharedPtr<sum_of_functions::Batch>());
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Batch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Batch_cClone
  (JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<lbfgs::Method, lbfgs::Batch, lbfgs::defaultDense>::getClone(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Batch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Batch_cGetInput
  (JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<lbfgs::Method, lbfgs::Batch, lbfgs::defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Batch
 * Method:    cGetParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Batch_cGetParameter
  (JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<lbfgs::Method, lbfgs::Batch, lbfgs::defaultDense>::getParameter(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Input
* Method:    cSetOptionalData
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Input_cSetOptionalData
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<lbfgs::Input>::set<lbfgs::OptionalDataId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Result
* Method:    cNewResult
* Signature: ()J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Result_cNewResult
(JNIEnv *, jobject)
{
    return jniArgument<lbfgs::Result>::newObj();
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Input
* Method:    cGetOptionalData
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Input_cGetOptionalData
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<lbfgs::Input>::get<lbfgs::OptionalDataId, NumericTable>(inputAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Result
* Method:    cGetOptionalData
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Result_cGetOptionalData
(JNIEnv *, jobject, jlong resAddr, jint id)
{
    return jniArgument<lbfgs::Result>::get<lbfgs::OptionalDataId, NumericTable>(resAddr, id);
}


/*
* Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Result
* Method:    cSetOptionalData
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Result_cSetOptionalData
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong ntAddr)
{
    jniArgument<lbfgs::Result>::set<lbfgs::OptionalDataId, NumericTable>(ntAddr, id, ntAddr);
}
