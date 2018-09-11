/* file: baseparameter.cpp */
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

#include "optimization_solver/sgd/JBaseParameter.h"

#include "common_defines.i"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cSetBatchIndices
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cSetBatchIndices
(JNIEnv *, jobject, jlong parAddr, jlong cBatchIndices)
{
    SerializationIfacePtr *ntShPtr = (SerializationIfacePtr *)cBatchIndices;
    ((sgd::BaseParameter *)parAddr)->batchIndices = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cGetBatchIndices
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cGetBatchIndices
(JNIEnv *, jobject, jlong parAddr)
{
    NumericTablePtr *ntShPtr = new NumericTablePtr();
    *ntShPtr = ((sgd::BaseParameter *)parAddr)->batchIndices;
    return (jlong)ntShPtr;
}


/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cSetLearningRateSequence
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cSetLearningRateSequence
(JNIEnv *, jobject, jlong parAddr, jlong cLearningRateSequence)
{
    SerializationIfacePtr *ntShPtr = (SerializationIfacePtr *)cLearningRateSequence;
    ((sgd::BaseParameter *)parAddr)->learningRateSequence = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cGetLearningRateSequence
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cGetLearningRateSequence
(JNIEnv *, jobject, jlong parAddr)
{
    NumericTablePtr *ntShPtr = new NumericTablePtr();
    *ntShPtr = ((sgd::BaseParameter *)parAddr)->learningRateSequence;
    return (jlong)ntShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cSetSeed
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cSetSeed
(JNIEnv *, jobject, jlong parAddr, jint seed)
{
    ((sgd::BaseParameter *)parAddr)->seed = seed;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cGetSeed
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cGetSeed
(JNIEnv *, jobject, jlong parAddr)
{
    return ((sgd::BaseParameter *)parAddr)->seed;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cSetEngine
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cSetEngine
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong engineAddr)
{
    (((sgd::BaseParameter *)cParameter))->engine = staticPointerCast<engines::BatchBase, AlgorithmIface> (*(SharedPtr<AlgorithmIface> *)engineAddr);
}
