/* file: tensor_impl.cpp */
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

#include "JTensorImpl.h"
#include "daal.h"

#include "java_tensor.h"

using namespace daal;
using namespace daal::services;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_management_data_TensorImpl
 * Method:    cAllocateDataMemory
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_TensorImpl_cAllocateDataMemory
  (JNIEnv * env, jobject thisObject, jlong cObject)
{
    Tensor *tensor = static_cast<Tensor *>(((SerializationIfacePtr *)cObject)->get());

    tensor->allocateDataMemory();

    if(tensor->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tensor->getErrors()->getDescription());
    }
}

/*
 * Class:     com_intel_daal_data_management_data_TensorImpl
 * Method:    cFreeDataMemory
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_TensorImpl_cFreeDataMemory
  (JNIEnv * env, jobject thisObject, jlong cObject)
{
    Tensor *tensor = static_cast<Tensor *>(((SerializationIfacePtr *)cObject)->get());

    tensor->freeDataMemory();

    if(tensor->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tensor->getErrors()->getDescription());
    }
}

/*
 * Class:     com_intel_daal_data_management_data_TensorImpl
 * Method:    cGetDimensions
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_data_1management_data_TensorImpl_cGetDimensions
  (JNIEnv * env, jobject thisObject, jlong cObject)
{
    Tensor *tensor = static_cast<Tensor *>(((SerializationIfacePtr *)cObject)->get());

    Collection<size_t> dims = tensor->getDimensions();

    if(tensor->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tensor->getErrors()->getDescription());
    }

    size_t size = dims.size();

    jlongArray jDims = env->NewLongArray(size);

    for(size_t i=0; i<size; i++)
    {
        jlong val = (jlong)dims[i];
        env->SetLongArrayRegion(jDims, i, 1, &val);
    }

    return jDims;
}

/*
 * Class:     com_intel_daal_data_management_data_TensorImpl
 * Method:    cSetDimensions
 * Signature: (J[J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_TensorImpl_cSetDimensions
  (JNIEnv * env, jobject thisObject, jlong cObject, jlongArray jDims)
{
    Tensor *tensor = static_cast<Tensor *>(((SerializationIfacePtr *)cObject)->get());

    jsize len   = env->GetArrayLength(jDims);
    jlong *dimSizes = env->GetLongArrayElements(jDims, 0);
    Collection<size_t> dims;
    for(size_t i=0; i<len; i++)
    {
        dims.push_back( dimSizes[i] );
    }
    env->ReleaseLongArrayElements(jDims, dimSizes, 0);

    tensor->setDimensions( dims );

    if(tensor->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tensor->getErrors()->getDescription());
    }
}

/*
 * Class:     com_intel_daal_data_management_data_TensorImpl
 * Method:    cNewJavaTensor
 * Signature: ([J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_TensorImpl_cNewJavaTensor
  (JNIEnv *env, jobject thisObj, jlongArray jDims)
{
    JavaVM *jvm;
    // Get pointer to the Java VM interface function table
    jint status = env->GetJavaVM(&jvm);
    if(status != 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), "Error on GetJavaVM");
        return 0;
    }

    jsize len   = env->GetArrayLength(jDims);
    jlong *dimSizes = env->GetLongArrayElements(jDims, 0);
    Collection<size_t> dims;
    for(size_t i=0; i<len; i++)
    {
        dims.push_back( dimSizes[i] );
    }
    env->ReleaseLongArrayElements(jDims, dimSizes, 0);

    // Create C++ object of the class Tensor
    Tensor* tnsr = new JavaTensor(dims, jvm, thisObj);
    if(tnsr->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tnsr->getErrors()->getDescription());
    }

    return (jlong)(new SerializationIfacePtr(tnsr));
}
