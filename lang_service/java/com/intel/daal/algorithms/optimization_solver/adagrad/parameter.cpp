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

#include "optimization_solver/adagrad/JParameter.h"

#include "common_defines.i"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::optimization_solver;
/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Parameter
 * Method:    cSetBatchIndices
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Parameter_cSetBatchIndices
(JNIEnv *, jobject, jlong parAddr, jlong cBatchIndices)
{
    SerializationIfacePtr *ntShPtr = (SerializationIfacePtr *)cBatchIndices;
    ((adagrad::Parameter *)parAddr)->batchIndices = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Parameter
 * Method:    cGetBatchIndices
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Parameter_cGetBatchIndices
(JNIEnv *, jobject, jlong parAddr)
{
    NumericTablePtr *ntShPtr = new NumericTablePtr();
    *ntShPtr = ((adagrad::Parameter *)parAddr)->batchIndices;
    return (jlong)ntShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Parameter
 * Method:    cSetBatchSize
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Parameter_cSetBatchSize
(JNIEnv *, jobject, jlong parAddr, jlong batchSize)
{
    ((adagrad::Parameter *)parAddr)->batchSize = batchSize;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Parameter
 * Method:    cGetBatchSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Parameter_cGetBatchSize
(JNIEnv *, jobject, jlong parAddr)
{
    return ((adagrad::Parameter *)parAddr)->batchSize;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Parameter
 * Method:    cSetLearningRate
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Parameter_cSetLearningRate
(JNIEnv *, jobject, jlong parAddr, jlong cLearningRate)
{
    SerializationIfacePtr *ntShPtr = (SerializationIfacePtr *)cLearningRate;
    ((adagrad::Parameter *)parAddr)->learningRate = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Parameter
 * Method:    cGetLearningRate
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Parameter_cGetLearningRate
(JNIEnv *, jobject, jlong parAddr)
{
    NumericTablePtr *ntShPtr = new NumericTablePtr();
    *ntShPtr = ((adagrad::Parameter *)parAddr)->learningRate;
    return (jlong)ntShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Parameter
 * Method:    cSetDegenerateCasesThreshold
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Parameter_cSetDegenerateCasesThreshold
(JNIEnv *, jobject, jlong parAddr, jdouble degenerateCasesThreshold)
{
    ((adagrad::Parameter *)parAddr)->degenerateCasesThreshold = degenerateCasesThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Parameter
 * Method:    cGetDegenerateCasesThreshold
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Parameter_cGetDegenerateCasesThreshold
(JNIEnv *, jobject, jlong parAddr)
{
    return ((adagrad::Parameter *)parAddr)->degenerateCasesThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Parameter
 * Method:    cSetSeed
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Parameter_cSetSeed
(JNIEnv *, jobject, jlong parAddr, jlong seed)
{
    ((adagrad::Parameter *)parAddr)->seed = seed;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Parameter
 * Method:    cGetSeed
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Parameter_cGetSeed
(JNIEnv *, jobject, jlong parAddr)
{
    return ((adagrad::Parameter *)parAddr)->seed;
}
