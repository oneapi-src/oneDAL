/* file: train_parameter.cpp */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
#include "com_intel_daal_algorithms_elastic_net_TrainParameter.h"
#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES();

/*
 * Class:     com_intel_daal_algorithms_elastic_net_training_TrainingParameter
 * Method:    cSetInterceptFlag
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_elastic_1net_training_TrainingParameter_cSetInterceptFlag(JNIEnv * env, jobject thisObj,
                                                                                                                jlong parAddr, jboolean flag)
{
    ((elastic_net::training::Parameter *)parAddr)->interceptFlag = flag;
}

/*
 * Class:     com_intel_daal_algorithms_elastic_net_training_TrainingParameter
 * Method:    cGetInterceptFlag
 * Signature:(J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_elastic_1net_training_TrainingParameter_cGetInterceptFlag(JNIEnv * env, jobject thisObj,
                                                                                                                    jlong parAddr)
{
    return ((elastic_net::training::Parameter *)parAddr)->interceptFlag;
}

/*
 * Class:     com_intel_daal_algorithms_elastic_net_TrainParameter
 * Method:    cSetPenaltyL1
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_elastic_1net_TrainParameter_cSetPenaltyL1(JNIEnv * env, jobject thisObj, jlong parAddr,
                                                                                                jlong cPenaltyL1)
{
    SerializationIfacePtr * const ntShPtr                    = (SerializationIfacePtr *)cPenaltyL1;
    ((elastic_net::training::Parameter *)parAddr)->penaltyL1 = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_elastic_net_TrainParameter
 * Method:    cSetPenaltyL2
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_elastic_1net_TrainParameter_cSetPenaltyL2(JNIEnv * env, jobject thisObj, jlong parAddr,
                                                                                                jlong cPenaltyL2)
{
    SerializationIfacePtr * const ntShPtr                    = (SerializationIfacePtr *)cPenaltyL2;
    ((elastic_net::training::Parameter *)parAddr)->penaltyL2 = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_elastic_net_TrainParameter
 * Method:    cGetPenaltyL1
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_elastic_1net_TrainParameter_cGetPenaltyL1(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    NumericTablePtr * const ntShPtr = new NumericTablePtr();
    *ntShPtr                        = ((elastic_net::training::Parameter *)parAddr)->penaltyL1;
    return (jlong)ntShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_elastic_net_TrainParameter
 * Method:    cGetPenaltyL2
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_elastic_1net_TrainParameter_cGetPenaltyL2(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    NumericTablePtr * const ntShPtr = new NumericTablePtr();
    *ntShPtr                        = ((elastic_net::training::Parameter *)parAddr)->penaltyL2;
    return (jlong)ntShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_elastic_net_TrainParameter
 * Method:    cSetDataUseInComputation
 * Signature:(JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_elastic_1net_TrainParameter_cSetDataUseInComputation(JNIEnv * env, jobject thisObj,
                                                                                                           jlong parAddr, jint flag)
{
    (*(elastic_net::training::Parameter *)parAddr).dataUseInComputation = (elastic_net::training::DataUseInComputation)flag;
}

/*
 * Class:     com_intel_daal_algorithms_elastic_net_TrainParameter
 * Method:    cGetDataUseInComputation
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_elastic_1net_TrainParameter_cGetDataUseInComputation(JNIEnv * env, jobject thisObj,
                                                                                                           jlong parAddr)
{
    return (jint)((*(elastic_net::training::Parameter *)parAddr).dataUseInComputation);
}

/*
 * Class:     com_intel_daal_algorithms_elastic_net_training_TrainingParameter
 * Method:    cSetOptResultToCompute
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_elastic_1net_training_TrainingParameter_cSetOptResultToCompute(JNIEnv * env, jobject thisObj,
                                                                                                                     jlong parAddr, jint optResult)
{
    (*(elastic_net::training::Parameter *)parAddr).optResultToCompute = optResult;
}

/*
 * Class:     com_intel_daal_algorithms_elastic_net_training_TrainingParameter
 * Method:    cGetOptResultToCompute
 * Signature:(J)Z
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_elastic_1net_training_TrainingParameter_cGetOptResultToCompute(JNIEnv * env, jobject thisObj,
                                                                                                                     jlong parAddr)
{
    return (*(elastic_net::training::Parameter *)parAddr).optResultToCompute;
}

/*
 * Class:     com_intel_daal_algorithms_elastic_net_training_TrainingParameter
 * Method:    cSetOptimizationSolver
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_elastic_1net_training_TrainingParameter_cSetOptimizationSolver(JNIEnv * env, jobject thisObj,
                                                                                                                     jlong parAddr,
                                                                                                                     jlong optimizationSolverAddr)
{
    (((elastic_net::training::Parameter *)parAddr))->optimizationSolver =
        staticPointerCast<optimization_solver::iterative_solver::Batch, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)optimizationSolverAddr);
}

/*
 * Class:     com_intel_daal_algorithms_elastic_net_training_TrainingParameter
 * Method:    cGetOptimizationSolver
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_elastic_1net_training_TrainingParameter_cGetOptimizationSolver(JNIEnv * env, jobject thisObj,
                                                                                                                      jlong parAddr)
{
    SharedPtr<optimization_solver::iterative_solver::Batch> * opt =
        new SharedPtr<optimization_solver::iterative_solver::Batch>((((elastic_net::training::Parameter *)parAddr))->optimizationSolver);
    return (jlong)opt;
}
