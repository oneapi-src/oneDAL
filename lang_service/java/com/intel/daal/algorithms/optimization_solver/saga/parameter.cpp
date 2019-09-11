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

#include "com_intel_daal_algorithms_optimization_solver_saga_Parameter.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::optimization_solver;
/*
 * Class:     com_intel_daal_algorithms_optimization_solver_saga_Parameter
 * Method:    cSetBatchIndices
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_saga_Parameter_cSetBatchIndices
(JNIEnv *, jobject, jlong parAddr, jlong cBatchIndices)
{
    SerializationIfacePtr *ntShPtr = (SerializationIfacePtr *)cBatchIndices;
    ((saga::Parameter *)parAddr)->batchIndices = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_saga_Parameter
 * Method:    cGetBatchIndices
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_saga_Parameter_cGetBatchIndices
(JNIEnv *, jobject, jlong parAddr)
{
    NumericTablePtr *ntShPtr = new NumericTablePtr();
    *ntShPtr = ((saga::Parameter *)parAddr)->batchIndices;
    return (jlong)ntShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_saga_Parameter
 * Method:    cSetLearningRate
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_saga_Parameter_cSetLearningRateSequence
(JNIEnv *, jobject, jlong parAddr, jlong cLearningRate)
{
    SerializationIfacePtr *ntShPtr = (SerializationIfacePtr *)cLearningRate;
    ((saga::Parameter *)parAddr)->learningRateSequence = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_saga_Parameter
 * Method:    cGetLearningRate
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_saga_Parameter_cGetLearningRateSequence
(JNIEnv *, jobject, jlong parAddr)
{
    NumericTablePtr *ntShPtr = new NumericTablePtr();
    *ntShPtr = ((saga::Parameter *)parAddr)->learningRateSequence;
    return (jlong)ntShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_saga_Parameter
 * Method:    cSetSeed
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_saga_Parameter_cSetSeed
(JNIEnv *, jobject, jlong parAddr, jlong seed)
{
    ((saga::Parameter *)parAddr)->seed = seed;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_saga_Parameter
 * Method:    cGetSeed
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_optimization_1solver_saga_Parameter_cGetSeed
(JNIEnv *, jobject, jlong parAddr)
{
    return ((saga::Parameter *)parAddr)->seed;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_saga_Parameter
 * Method:    cSetEngine
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_saga_Parameter_cSetEngine
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong engineAddr)
{
    (((saga::Parameter *)cParameter))->engine = staticPointerCast<engines::BatchBase, AlgorithmIface> (*(SharedPtr<AlgorithmIface> *)engineAddr);
}
