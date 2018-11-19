/* file: kernelfunction_rbf_parameter.cpp */
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
#include "rbf/JParameter.h"
#include "daal.h"

using namespace daal::algorithms::kernel_function::rbf;
/*
 * Class:     com_intel_daal_algorithms_kernel_function_rbf_Parameter
 * Method:    cSetSigma
 * Signature:(D)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kernel_1function_rbf_Parameter_cSetSigma
(JNIEnv *env, jobject thisObj, jlong parAddr, jdouble sigma)
{
    (*(Parameter *)parAddr).sigma = sigma;
}
/*
 * Class:     com_intel_daal_algorithms_kernel_function_rbf_Parameter
 * Method:    cGetSigma
 * Signature:(D)J
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Parameter_cGetSigma
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return(jdouble)(*(Parameter *)parAddr).sigma;
}
