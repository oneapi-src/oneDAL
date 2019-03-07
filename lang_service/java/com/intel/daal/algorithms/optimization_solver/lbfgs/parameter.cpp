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

#include "optimization_solver/lbfgs/JParameter.h"

#include "common_defines.i"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
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

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_lbfgs_Parameter
 * Method:    cSetEngine
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_lbfgs_Parameter_cSetEngine
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong engineAddr)
{
    (((lbfgs::Parameter *)cParameter))->engine = staticPointerCast<engines::BatchBase, AlgorithmIface> (*(SharedPtr<AlgorithmIface> *)engineAddr);
}
