/* file: train_parameter.cpp */
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
#include "com_intel_daal_algorithms_lasso_regression_TrainParameter.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();

/*
 * Class:     com_intel_daal_algorithms_lasso_regression_training_TrainingParameter
 * Method:    cSetInterceptFlag
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_lasso_1regression_training_TrainingParameter_cSetInterceptFlag
(JNIEnv *env, jobject thisObj, jlong parAddr, jboolean flag)
{
    ((lasso_regression::training::Parameter *)parAddr)->interceptFlag = flag;
}

/*
 * Class:     com_intel_daal_algorithms_lasso_regression_training_TrainingParameter
 * Method:    cGetInterceptFlag
 * Signature:(J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_lasso_1regression_training_TrainingParameter_cGetInterceptFlag
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return((lasso_regression::training::Parameter *)parAddr)->interceptFlag;
}

/*
 * Class:     com_intel_daal_algorithms_lasso_regression_TrainParameter
 * Method:    cSetLassoParameters
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_lasso_1regression_TrainParameter_cSetLassoParameters
(JNIEnv * env, jobject thisObj, jlong parAddr, jlong cLassoParameters)
{
    SerializationIfacePtr * const ntShPtr = (SerializationIfacePtr *)cLassoParameters;
    ((lasso_regression::training::Parameter *)parAddr)->lassoParameters = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_lasso_regression_TrainParameter
 * Method:    cGetLassoParameters
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_lasso_1regression_TrainParameter_cGetLassoParameters
(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    NumericTablePtr * const ntShPtr = new NumericTablePtr();
    *ntShPtr = ((lasso_regression::training::Parameter *)parAddr)->lassoParameters;
    return (jlong)ntShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_lasso_1regression_TrainParameter
 * Method:    cSetDataUseInComputation
 * Signature:(JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_lasso_1regression_TrainParameter_cSetDataUseInComputation
(JNIEnv *env, jobject thisObj, jlong parAddr, jint flag)
{
    (*(lasso_regression::training::Parameter *)parAddr).dataUseInComputation = (lasso_regression::training::DataUseInComputation)flag;
}

/*
 * Class:     com_intel_daal_algorithms_lasso_1regression_TrainParameter
 * Method:    cGetDataUseInComputation
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_lasso_1regression_TrainParameter_cGetDataUseInComputation
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return (jint)((*(lasso_regression::training::Parameter *)parAddr).dataUseInComputation);
}

/*
 * Class:     com_intel_daal_algorithms_lasso_regression_training_TrainingParameter
 * Method:    cSetOptResultToCompute
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_lasso_1regression_training_TrainingParameter_cSetOptResultToCompute
(JNIEnv *env, jobject thisObj, jlong parAddr, jint optResult)
{
    (*(lasso_regression::training::Parameter *)parAddr).optResultToCompute = optResult;
}

/*
 * Class:     com_intel_daal_algorithms_lasso_regression_training_TrainingParameter
 * Method:    cGetOptResultToCompute
 * Signature:(J)Z
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_lasso_1regression_training_TrainingParameter_cGetOptResultToCompute
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return(*(lasso_regression::training::Parameter *)parAddr).optResultToCompute;
}


/*
 * Class:     com_intel_daal_algorithms_lasso_regression_training_TrainingParameter
 * Method:    cSetOptimizationSolver
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_lasso_1regression_training_TrainingParameter_cSetOptimizationSolver
(JNIEnv *env, jobject thisObj, jlong parAddr, jlong optimizationSolverAddr)
{
    (((lasso_regression::training::Parameter *)parAddr))->optimizationSolver = staticPointerCast<optimization_solver::iterative_solver::Batch, AlgorithmIface> (*(SharedPtr<AlgorithmIface> *)optimizationSolverAddr);
}

/*
 * Class:     com_intel_daal_algorithms_lasso_regression_training_TrainingParameter
 * Method:    cGetOptimizationSolver
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_lasso_1regression_training_TrainingParameter_cGetOptimizationSolver
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    SharedPtr<optimization_solver::iterative_solver::Batch > *opt =
        new SharedPtr<optimization_solver::iterative_solver::Batch >((((lasso_regression::training::Parameter *)parAddr))->optimizationSolver);
    return (jlong)opt;
}
