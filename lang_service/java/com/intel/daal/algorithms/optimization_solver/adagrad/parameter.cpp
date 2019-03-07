/* file: parameter.cpp */
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

#include "optimization_solver/adagrad/JParameter.h"

#include "common_defines.i"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
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

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Parameter
 * Method:    cSetEngine
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_adagrad_Parameter_cSetEngine
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong engineAddr)
{
    (((adagrad::Parameter *)cParameter))->engine = staticPointerCast<engines::BatchBase, AlgorithmIface> (*(SharedPtr<AlgorithmIface> *)engineAddr);
}
