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

#include "optimization_solver/lbfgs/JParameter.h"

#include "common_defines.i"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cSetM
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cSetM
  (JNIEnv *, jobject, jlong parAddr, jlong m)
{
    ((lbfgs::Parameter *)parAddr)->m = (size_t)m;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cGetM
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cGetM
  (JNIEnv *, jobject, jlong parAddr)
{
    return (jlong)(((lbfgs::Parameter *)parAddr)->m);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cSetL
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cSetL
  (JNIEnv *, jobject, jlong parAddr, jlong L)
{
    ((lbfgs::Parameter *)parAddr)->L = (size_t)L;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cGetL
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cGetL
  (JNIEnv *, jobject, jlong parAddr)
{
    return (jlong)(((lbfgs::Parameter *)parAddr)->L);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cSetBatchSize
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cSetBatchSize
  (JNIEnv *, jobject, jlong parAddr, jlong batchSize)
{
    ((lbfgs::Parameter *)parAddr)->batchSize = (size_t)batchSize;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cGetBatchSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cGetBatchSize
  (JNIEnv *, jobject, jlong parAddr)
{
    return (jlong)(((lbfgs::Parameter *)parAddr)->batchSize);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cSetBatchIndices
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cSetBatchIndices
  (JNIEnv *, jobject, jlong parAddr, jlong ntAddr)
{
    SerializationIfacePtr *ntShPtr = (SerializationIfacePtr *)ntAddr;
    ((lbfgs::Parameter *)parAddr)->batchIndices = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cGetBatchIndices
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cGetBatchIndices
  (JNIEnv *, jobject, jlong parAddr)
{
    SerializationIfacePtr *ntShPtr = new SerializationIfacePtr();
    *ntShPtr = ((lbfgs::Parameter *)parAddr)->batchIndices;
    return (jlong)ntShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cSetCorrectionPairBatchSize
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cSetCorrectionPairBatchSize
  (JNIEnv *, jobject, jlong parAddr, jlong batchSize)
{
    ((lbfgs::Parameter *)parAddr)->correctionPairBatchSize = (size_t)batchSize;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cGetCorrectionPairBatchSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cGetCorrectionPairBatchSize
  (JNIEnv *, jobject, jlong parAddr)
{
    return (jlong)(((lbfgs::Parameter *)parAddr)->correctionPairBatchSize);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cSetCorrectionPairBatchIndices
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cSetCorrectionPairBatchIndices
  (JNIEnv *, jobject, jlong parAddr, jlong ntAddr)
{
    SerializationIfacePtr *ntShPtr = (SerializationIfacePtr *)ntAddr;
    ((lbfgs::Parameter *)parAddr)->correctionPairBatchIndices =
        staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cGetCorrectionPairBatchIndices
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cGetCorrectionPairBatchIndices
  (JNIEnv *, jobject, jlong parAddr)
{
    SerializationIfacePtr *ntShPtr = new SerializationIfacePtr();
    *ntShPtr = ((lbfgs::Parameter *)parAddr)->correctionPairBatchIndices;
    return (jlong)ntShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cSetStepLengthSequence
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cSetStepLengthSequence
  (JNIEnv *, jobject, jlong parAddr, jlong ntAddr)
{
    SerializationIfacePtr *ntShPtr = (SerializationIfacePtr *)ntAddr;
    ((lbfgs::Parameter *)parAddr)->stepLengthSequence =
        staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cGetStepLengthSequence
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cGetStepLengthSequence
  (JNIEnv *, jobject, jlong parAddr)
{
    SerializationIfacePtr *ntShPtr = new SerializationIfacePtr();
    *ntShPtr = ((lbfgs::Parameter *)parAddr)->stepLengthSequence;
    return (jlong)ntShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cSetSeed
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cSetSeed
  (JNIEnv *, jobject, jlong parAddr, jlong seed)
{
    ((lbfgs::Parameter *)parAddr)->seed = (size_t)seed;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cGetSeed
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cGetSeed
  (JNIEnv *, jobject, jlong parAddr)
{
    return ((lbfgs::Parameter *)parAddr)->seed;
}
