/* file: kernelfunction_linear.cpp */
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
#include "linear/JBatch.h"
#include "linear/JResult.h"
#include "linear/JMethod.h"
#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::kernel_function::linear;

#define DefaultDense com_intel_daal_algorithms_kernel_function_linear_Method_defaultDenseValue
#define FastCSR      com_intel_daal_algorithms_kernel_function_linear_Method_fastCSRValue

/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Batch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Batch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<kernel_function::linear::Method, Batch, defaultDense, fastCSR>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Batch
 * Method:    cGetParameter
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Batch_cGetParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<kernel_function::linear::Method, Batch, defaultDense, fastCSR>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Batch
 * Method:    cSetResult
 * Signature:(JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Batch_cSetResult
(JNIEnv *env, jobject obj, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniBatch<kernel_function::linear::Method, Batch, defaultDense, fastCSR>::setResult<kernel_function::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Batch
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Batch_cGetResult
(JNIEnv *env, jobject obj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<kernel_function::linear::Method, Batch, defaultDense, fastCSR>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Batch
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Batch_cGetInput
(JNIEnv *env, jobject obj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<kernel_function::linear::Method, Batch, defaultDense, fastCSR>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Batch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Batch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<kernel_function::linear::Method, Batch, defaultDense, fastCSR>::getClone(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Batch_Result
 * Method:    cGetResult
 * Signature:()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Result_cNewResult
(JNIEnv *env, jobject obj)
{
    return jniArgument<kernel_function::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Batch_Result
 * Method:    cGetResultTable
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Result_cGetResultTable
(JNIEnv *env, jobject obj, jlong resAddr, jint id)
{
    return jniArgument<kernel_function::Result>::get<kernel_function::ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_kernel_function_linear_Batch_Result
 * Method:    cSetResultTable
 * Signature:(JJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kernel_1function_linear_Result_cSetResultTable
(JNIEnv *env, jobject obj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<kernel_function::Result>::set<kernel_function::ResultId, NumericTable>(resAddr, id, ntAddr);
}
