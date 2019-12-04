/* file: training_parameter.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#include "com_intel_daal_algorithms_logistic_regression_training_TrainingParameter.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_training_TrainingParameter
 * Method:    cSetInterceptFlag
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logistic_1regression_training_TrainingParameter_cSetInterceptFlag(JNIEnv * env, jobject thisObj,
                                                                                                                        jlong algAddr, jboolean flag)
{
    (*(logistic_regression::training::Parameter *)algAddr).interceptFlag = flag;
}

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_training_TrainingParameter
 * Method:    cGetInterceptFlag
 * Signature:(J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_logistic_1regression_training_TrainingParameter_cGetInterceptFlag(JNIEnv * env,
                                                                                                                            jobject thisObj,
                                                                                                                            jlong algAddr)
{
    return (*(logistic_regression::training::Parameter *)algAddr).interceptFlag;
}

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_training_TrainingParameter
 * Method:    cSetPenaltyL1
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logistic_1regression_training_TrainingParameter_cSetPenaltyL1(JNIEnv * env, jobject thisObj,
                                                                                                                    jlong algAddr, jfloat penaltyL1)
{
    (*(logistic_regression::training::Parameter *)algAddr).penaltyL1 = penaltyL1;
}

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_training_TrainingParameter
 * Method:    cGetPenaltyL1
 * Signature:(J)Z
 */
JNIEXPORT jfloat JNICALL Java_com_intel_daal_algorithms_logistic_1regression_training_TrainingParameter_cGetPenaltyL1(JNIEnv * env, jobject thisObj,
                                                                                                                      jlong algAddr)
{
    return (*(logistic_regression::training::Parameter *)algAddr).penaltyL1;
}

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_training_TrainingParameter
 * Method:    cSetPenaltyL2
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logistic_1regression_training_TrainingParameter_cSetPenaltyL2(JNIEnv * env, jobject thisObj,
                                                                                                                    jlong algAddr, jfloat penaltyL2)
{
    (*(logistic_regression::training::Parameter *)algAddr).penaltyL2 = penaltyL2;
}

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_training_TrainingParameter
 * Method:    cGetPenaltyL2
 * Signature:(J)Z
 */
JNIEXPORT jfloat JNICALL Java_com_intel_daal_algorithms_logistic_1regression_training_TrainingParameter_cGetPenaltyL2(JNIEnv * env, jobject thisObj,
                                                                                                                      jlong algAddr)
{
    return (*(logistic_regression::training::Parameter *)algAddr).penaltyL2;
}

/*
 * Class:     com_intel_daal_algorithms_decision_forest_regression_training_TrainingParameter
 * Method:    cSetOptimizationSolver
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_decision_1forest_regression_training_TrainingParameter_cSetOptimizationSolver(
    JNIEnv * env, jobject thisObj, jlong cTrainingParameter, jlong optimizationSolverAddr)
{
    (((logistic_regression::training::Parameter *)cTrainingParameter))->optimizationSolver =
        staticPointerCast<optimization_solver::iterative_solver::Batch, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)optimizationSolverAddr);
}
