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

#include "optimization_solver/sgd/JParameterMiniBatch.h"
#include "optimization_solver/sgd/JParameterMomentum.h"

#include "common_defines.i"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::optimization_solver;

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

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_ParameterMomentum
 * Method:    cSetMomentum
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_ParameterMomentum_cSetMomentum
(JNIEnv *, jobject, jlong parAddr, jdouble momentum)
{
    ((sgd::Parameter<sgd::momentum> *)parAddr)->momentum = momentum;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_ParameterMomentum
 * Method:    cGetMomentum
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_ParameterMomentum_cGetMomentum
(JNIEnv *, jobject, jlong parAddr)
{
    return ((sgd::Parameter<sgd::momentum> *)parAddr)->momentum;
}
