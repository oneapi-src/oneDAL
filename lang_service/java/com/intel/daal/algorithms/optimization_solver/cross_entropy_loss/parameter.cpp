/* file: parameter.cpp */
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

#include "com_intel_daal_algorithms_optimization_solver_cross_entropy_loss_Parameter.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_cross_entropy_loss_Parameter
 * Method:    cSetInterceptFlag
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Parameter_cSetInterceptFlag(JNIEnv * env,
                                                                                                                            jobject thisObj,
                                                                                                                            jlong algAddr,
                                                                                                                            jboolean flag)
{
    (*(cross_entropy_loss::Parameter *)algAddr).interceptFlag = flag;
}

/*
 * Class:     com_intel_daal_algorithms_cross_entropy_loss_Parameter
 * Method:    cGetInterceptFlag
 * Signature:(J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Parameter_cGetInterceptFlag(JNIEnv * env,
                                                                                                                                jobject thisObj,
                                                                                                                                jlong algAddr)
{
    return (*(cross_entropy_loss::Parameter *)algAddr).interceptFlag;
}

/*
 * Class:     com_intel_daal_algorithms_cross_entropy_loss_Parameter
 * Method:    cSetPenaltyL1
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Parameter_cSetPenaltyL1(JNIEnv * env, jobject thisObj,
                                                                                                                        jlong algAddr,
                                                                                                                        jfloat penaltyL1)
{
    (*(cross_entropy_loss::Parameter *)algAddr).penaltyL1 = penaltyL1;
}

/*
 * Class:     com_intel_daal_algorithms_cross_entropy_loss_Parameter
 * Method:    cGetPenaltyL1
 * Signature:(J)Z
 */
JNIEXPORT jfloat JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Parameter_cGetPenaltyL1(JNIEnv * env,
                                                                                                                          jobject thisObj,
                                                                                                                          jlong algAddr)
{
    return (*(cross_entropy_loss::Parameter *)algAddr).penaltyL1;
}

/*
 * Class:     com_intel_daal_algorithms_cross_entropy_loss_Parameter
 * Method:    cSetPenaltyL2
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Parameter_cSetPenaltyL2(JNIEnv * env, jobject thisObj,
                                                                                                                        jlong algAddr,
                                                                                                                        jfloat penaltyL2)
{
    (*(cross_entropy_loss::Parameter *)algAddr).penaltyL2 = penaltyL2;
}

/*
 * Class:     com_intel_daal_algorithms_cross_entropy_loss_Parameter
 * Method:    cGetPenaltyL2
 * Signature:(J)Z
 */
JNIEXPORT jfloat JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Parameter_cGetPenaltyL2(JNIEnv * env,
                                                                                                                          jobject thisObj,
                                                                                                                          jlong algAddr)
{
    return (*(cross_entropy_loss::Parameter *)algAddr).penaltyL2;
}

/*
 * Class:     com_intel_daal_algorithms_cross_entropy_loss_Parameter
 * Method:    cSetNClasses
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Parameter_cSetNClasses(JNIEnv * env, jobject thisObj,
                                                                                                                       jlong algAddr, jlong nClasses)
{
    (*(cross_entropy_loss::Parameter *)algAddr).nClasses = nClasses;
}

/*
 * Class:     com_intel_daal_algorithms_cross_entropy_loss_Parameter
 * Method:    cGetNClasses
 * Signature:(J)Z
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Parameter_cGetNClasses(JNIEnv * env, jobject thisObj,
                                                                                                                        jlong algAddr)
{
    return (*(cross_entropy_loss::Parameter *)algAddr).nClasses;
}
