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

#include "optimization_solver/saga/JBatch.h"
#include "optimization_solver/saga/JInput.h"
#include "optimization_solver/saga/JResult.h"

#include "common_defines.i"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_saga_Batch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_saga_Batch_cInit
(JNIEnv *, jobject, jint prec, jint method)
{
    return jniBatch<saga::Method, saga::Batch, saga::defaultDense>::newObj(prec, method, SharedPtr<sum_of_functions::Batch>());
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_saga_Batch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_saga_Batch_cClone
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<saga::Method, saga::Batch, saga::defaultDense>::getClone(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_saga_Batch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_saga_Batch_cGetInput
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<saga::Method, saga::Batch, saga::defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_saga_Batch
 * Method:    cGetParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_saga_Batch_cGetParameter
(JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<saga::Method, saga::Batch, saga::defaultDense>::getParameter(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_saga_Input
* Method:    cSetOptionalData
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_saga_Input_cSetOptionalData
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<saga::Input>::set<saga::OptionalDataId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_saga_Result
* Method:    cNewResult
* Signature: ()J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_saga_Result_cNewResult
(JNIEnv *, jobject)
{
    return jniArgument<saga::Result>::newObj();
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_saga_Input
* Method:    cGetOptionalData
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_saga_Input_cGetOptionalData
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<saga::Input>::get<saga::OptionalDataId, NumericTable>(inputAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_saga_Result
* Method:    cGetOptionalData
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_saga_Result_cGetOptionalData
(JNIEnv *, jobject, jlong resAddr, jint id)
{
    return jniArgument<saga::Result>::get<saga::OptionalDataId, NumericTable>(resAddr, id);
}


/*
* Class:     com_intel_daal_algorithms_optimization_solver_saga_Result
* Method:    cSetOptionalData
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_saga_Result_cSetOptionalData
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong ntAddr)
{
    jniArgument<saga::Result>::set<saga::OptionalDataId, NumericTable>(ntAddr, id, ntAddr);
}
