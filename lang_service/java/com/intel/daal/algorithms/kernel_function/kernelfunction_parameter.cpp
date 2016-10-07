/* file: kernelfunction_parameter.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
#include "kernel_function/JParameter.h"
#include "kernel_function/JComputationMode.h"
#include "daal.h"

#define VectorVector com_intel_daal_algorithms_kernel_function_ComputationMode_VectorVector
#define MatrixVector com_intel_daal_algorithms_kernel_function_ComputationMode_MatrixVector
#define MatrixMatrix com_intel_daal_algorithms_kernel_function_ComputationMode_MatrixMatrix

using namespace daal::algorithms::kernel_function;
/*
 * Class:     com_intel_daal_algorithms_kernel_function_Parameter
 * Method:    init
 * Signature:(JJJI)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kernel_1function_Parameter_cSetComputationMode
(JNIEnv *env, jobject thisObj, jlong parAddr, jint id)
{
    if(id == VectorVector)
    {
        (*(ParameterBase *)parAddr).computationMode = vectorVector;
    }
    else if(id == MatrixVector)
    {
        (*(ParameterBase *)parAddr).computationMode = matrixVector;
    }
    else if(id == MatrixMatrix)
    {
        (*(ParameterBase *)parAddr).computationMode = matrixMatrix;
    }
}
/*
 * Class:     com_intel_daal_algorithms_kernel_function_Parameter
 * Method:    init
 * Signature:(JJJI)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kernel_1function_Parameter_cSetRowIndexX
(JNIEnv *env, jobject thisObj, jlong parAddr, jlong indexX)
{
    (*(ParameterBase *)parAddr).rowIndexX = indexX;
}
/*
 * Class:     com_intel_daal_algorithms_kernel_function_Parameter
 * Method:    cGetRowIndexX
 * Signature:(JJJI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kernel_1function_Parameter_cGetRowIndexX
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return(jlong)(*(ParameterBase *)parAddr).rowIndexX;
}

/*
 * Class:     com_intel_daal_algorithms_kernel_function_Parameter
 * Method:    cSetRowIndexY
 * Signature:(JJJI)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kernel_1function_Parameter_cSetRowIndexY
(JNIEnv *env, jobject thisObj, jlong parAddr, jlong indexY)
{
    (*(ParameterBase *)parAddr).rowIndexY = indexY;
}

/*
 * Class:     com_intel_daal_algorithms_kernel_function_Parameter
 * Method:    cGetRowIndexX
 * Signature:(JJJI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kernel_1function_Parameter_cGetRowIndexY
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return(jlong)(*(ParameterBase *)parAddr).rowIndexY;
}
/*
 * Class:     com_intel_daal_algorithms_kernel_function_Parameter
 * Method:    cSetRowIndexResult
 * Signature:(JJJI)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kernel_1function_Parameter_cSetRowIndexResult
(JNIEnv *env, jobject thisObj, jlong parAddr, jlong indexResult)
{
    (*(ParameterBase *)parAddr).rowIndexResult = indexResult;
}
/*
 * Class:     com_intel_daal_algorithms_kernel_function_Parameter
 * Method:    cGetRowIndexResult
 * Signature:(JJJI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kernel_1function_Parameter_cGetRowIndexResult
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return(jlong)(*(ParameterBase *)parAddr).rowIndexResult;
}
