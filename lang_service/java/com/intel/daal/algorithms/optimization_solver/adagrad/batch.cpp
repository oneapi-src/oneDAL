/* file: batch.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
