/* file: parameter.cpp */
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

#include "optimization_solver/cross_entropy_loss/JParameter.h"

#include "common_defines.i"

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
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Parameter_cSetInterceptFlag
(JNIEnv *env, jobject thisObj, jlong algAddr, jboolean flag)
{
    (*(cross_entropy_loss::Parameter *)algAddr).interceptFlag = flag;
}

/*
 * Class:     com_intel_daal_algorithms_cross_entropy_loss_Parameter
 * Method:    cGetInterceptFlag
 * Signature:(J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Parameter_cGetInterceptFlag
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    return(*(cross_entropy_loss::Parameter *)algAddr).interceptFlag;
}

/*
 * Class:     com_intel_daal_algorithms_cross_entropy_loss_Parameter
 * Method:    cSetPenaltyL1
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Parameter_cSetPenaltyL1
(JNIEnv *env, jobject thisObj, jlong algAddr, jfloat penaltyL1)
{
    (*(cross_entropy_loss::Parameter *)algAddr).penaltyL1 = penaltyL1;
}

/*
 * Class:     com_intel_daal_algorithms_cross_entropy_loss_Parameter
 * Method:    cGetPenaltyL1
 * Signature:(J)Z
 */
JNIEXPORT jfloat JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Parameter_cGetPenaltyL1
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    return(*(cross_entropy_loss::Parameter *)algAddr).penaltyL1;
}

/*
 * Class:     com_intel_daal_algorithms_cross_entropy_loss_Parameter
 * Method:    cSetPenaltyL2
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Parameter_cSetPenaltyL2
(JNIEnv *env, jobject thisObj, jlong algAddr, jfloat penaltyL2)
{
    (*(cross_entropy_loss::Parameter *)algAddr).penaltyL2 = penaltyL2;
}

/*
 * Class:     com_intel_daal_algorithms_cross_entropy_loss_Parameter
 * Method:    cGetPenaltyL2
 * Signature:(J)Z
 */
JNIEXPORT jfloat JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Parameter_cGetPenaltyL2
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    return(*(cross_entropy_loss::Parameter *)algAddr).penaltyL2;
}

/*
 * Class:     com_intel_daal_algorithms_cross_entropy_loss_Parameter
 * Method:    cSetNClasses
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Parameter_cSetNClasses
(JNIEnv *env, jobject thisObj, jlong algAddr, jlong nClasses)
{
    (*(cross_entropy_loss::Parameter *)algAddr).nClasses = nClasses;
}

/*
 * Class:     com_intel_daal_algorithms_cross_entropy_loss_Parameter
 * Method:    cGetNClasses
 * Signature:(J)Z
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_cross_1entropy_1loss_Parameter_cGetNClasses
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    return(*(cross_entropy_loss::Parameter *)algAddr).nClasses;
}
