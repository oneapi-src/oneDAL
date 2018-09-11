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

#include "optimization_solver/adagrad/JBatch.h"
#include "optimization_solver/adagrad/JInput.h"
#include "optimization_solver/adagrad/JResult.h"

#include "common_defines.i"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Batch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Batch_cInit
(JNIEnv *, jobject, jint prec, jint method)
{
    return jniBatch<adagrad::Method, adagrad::Batch, adagrad::defaultDense>::newObj(prec, method, SharedPtr<sum_of_functions::Batch>());
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Batch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Batch_cClone
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<adagrad::Method, adagrad::Batch, adagrad::defaultDense>::getClone(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Batch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Batch_cGetInput
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<adagrad::Method, adagrad::Batch, adagrad::defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Batch
 * Method:    cGetParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Batch_cGetParameter
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<adagrad::Method, adagrad::Batch, adagrad::defaultDense>::getParameter(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Input
* Method:    cSetOptionalData
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Input_cSetOptionalData
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<adagrad::Input>::set<adagrad::OptionalDataId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Result
* Method:    cNewResult
* Signature: ()J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Result_cNewResult
(JNIEnv *, jobject)
{
    return jniArgument<adagrad::Result>::newObj();
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Input
* Method:    cGetOptionalData
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Input_cGetOptionalData
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<adagrad::Input>::get<adagrad::OptionalDataId, NumericTable>(inputAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Result
* Method:    cGetOptionalData
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Result_cGetOptionalData
(JNIEnv *, jobject, jlong resAddr, jint id)
{
    return jniArgument<adagrad::Result>::get<adagrad::OptionalDataId, NumericTable>(resAddr, id);
}


/*
* Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Result
* Method:    cSetOptionalData
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Result_cSetOptionalData
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong ntAddr)
{
    jniArgument<adagrad::Result>::set<adagrad::OptionalDataId, NumericTable>(ntAddr, id, ntAddr);
}
