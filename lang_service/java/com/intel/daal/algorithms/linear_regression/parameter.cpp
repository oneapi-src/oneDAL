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
#include "linear_regression/JParameter.h"

using namespace daal;
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_linear_regression_Parameter
 * Method:    cParInit
 * Signature:()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_Parameter_cParInit
(JNIEnv *env, jobject thisObj)
{
    return(jlong)new linear_regression::Parameter();
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_Parameter
 * Method:    cSetInterceptFlag
 * Signature:(JZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_linear_1regression_Parameter_cSetInterceptFlag
(JNIEnv *env, jobject thisObj, jlong algAddr, jboolean flag)
{
    (*(linear_regression::Parameter *)algAddr).interceptFlag = flag;
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_Parameter
 * Method:    cGetInterceptFlag
 * Signature:(J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_linear_1regression_Parameter_cGetInterceptFlag
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    return(*(linear_regression::Parameter *)algAddr).interceptFlag;
}
