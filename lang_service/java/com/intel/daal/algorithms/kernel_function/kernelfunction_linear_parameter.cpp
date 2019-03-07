/* file: kernelfunction_linear_parameter.cpp */
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
#include "linear/JParameter.h"
#include "daal.h"

using namespace daal::algorithms::kernel_function::linear;
/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Parameter
 * Method:    cSetK
 * Signature:(DD)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Parameter_cSetK
(JNIEnv *env, jobject thisObj, jlong parAddr, jdouble k)
{
    (*(Parameter *)parAddr).k = k;
}
/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Parameter
 * Method:    cSetK
 * Signature:(DD)J
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Parameter_cGetK
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return(jdouble)(*(Parameter *)parAddr).k;
}

/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Parameter
 * Method:    cSetK
 * Signature:(DD)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Parameter_cSetB
(JNIEnv *env, jobject thisObj, jlong parAddr, jdouble b)
{
    (*(Parameter *)parAddr).b = b;
}
/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Parameter
 * Method:    cGetB
 * Signature:(DD)J
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Parameter_cGetB
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return(jdouble)(*(Parameter *)parAddr).b;
}
