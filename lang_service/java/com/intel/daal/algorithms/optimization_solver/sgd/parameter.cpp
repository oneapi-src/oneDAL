/* file: parameter.cpp */
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

#include "optimization_solver/sgd/JParameterMiniBatch.h"

#include "common_defines.i"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_ParameterMiniBatch
 * Method:    cSetBatchSize
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_ParameterMiniBatch_cSetBatchSize
(JNIEnv *, jobject, jlong parAddr, jlong batchSize)
{
    ((sgd::Parameter<sgd::miniBatch> *)parAddr)->batchSize = batchSize;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_ParameterMiniBatch
 * Method:    cGetBatchSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_ParameterMiniBatch_cGetBatchSize
(JNIEnv *, jobject, jlong parAddr)
{
    return ((sgd::Parameter<sgd::miniBatch> *)parAddr)->batchSize;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_ParameterMiniBatch
 * Method:    cSetInnerNIterations
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_ParameterMiniBatch_cSetInnerNIterations
(JNIEnv *, jobject, jlong parAddr, jlong innerNIterations)
{
    ((sgd::Parameter<sgd::miniBatch> *)parAddr)->innerNIterations = innerNIterations;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_ParameterMiniBatch
 * Method:    cGetInnerNIterations
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_ParameterMiniBatch_cGetInnerNIterations
(JNIEnv *, jobject, jlong parAddr)
{
    return ((sgd::Parameter<sgd::miniBatch> *)parAddr)->innerNIterations;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_ParameterMiniBatch
 * Method:    cSetConservativeSequence
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_ParameterMiniBatch_cSetConservativeSequence
(JNIEnv *, jobject, jlong parAddr, jlong cConservativeSequence)
{
    SerializationIfacePtr *ntShPtr = (SerializationIfacePtr *)cConservativeSequence;
    ((sgd::Parameter<sgd::miniBatch> *)parAddr)->conservativeSequence = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_ParameterMiniBatch
 * Method:    cGetConservativeSequence
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_ParameterMiniBatch_cGetConservativeSequence
(JNIEnv *, jobject, jlong parAddr)
{
    NumericTablePtr *ntShPtr = new NumericTablePtr();
    *ntShPtr = ((sgd::Parameter<sgd::miniBatch> *)parAddr)->conservativeSequence;
    return (jlong)ntShPtr;
}
