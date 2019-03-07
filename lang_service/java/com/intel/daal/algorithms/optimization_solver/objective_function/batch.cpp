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

#include "optimization_solver/objective_function/JBatch.h"

#include "common_defines.i"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_objective_function_Batch
 * Method:    cSetResult
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_objective_1function_Batch_cSetResult
(JNIEnv *, jobject, jlong algAddr, jlong cResult)
{
    SharedPtr<objective_function::Batch> alg =
        staticPointerCast<objective_function::Batch, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);

    SerializationIfacePtr *serializableShPtr = (SerializationIfacePtr *)cResult;
    objective_function::ResultPtr resultShPtr =
        staticPointerCast<objective_function::Result, SerializationIface>(*serializableShPtr);

    alg->setResult(resultShPtr);
}



/*
 * Class:     com_intel_daal_algorithms_optimization_solver_objective_function_Batch
 * Method:    cGetResult
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_objective_1function_Batch_cGetResult
(JNIEnv *, jobject, jlong algAddr)
{
    SharedPtr<objective_function::Batch> alg =
        staticPointerCast<objective_function::Batch, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);

    SerializationIfacePtr *resShPtr = new SerializationIfacePtr();
    *resShPtr = staticPointerCast<SerializationIface, objective_function::Result>(alg->getResult());
    return (jlong)(resShPtr);
}
