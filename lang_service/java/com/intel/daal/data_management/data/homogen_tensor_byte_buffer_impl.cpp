/* file: homogen_tensor_byte_buffer_impl.cpp */
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

#include "JHomogenTensorByteBufferImpl.h"
#include "daal.h"
#include "common_helpers_functions.h"

using namespace daal;
using namespace daal::services;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_management_data_HomogenTensorByteBufferImpl
 * Method:    getIndexType
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_HomogenTensorByteBufferImpl_getIndexType
  (JNIEnv * env, jobject thisObject, jlong cObject)
{
    Tensor *tensor = ((TensorPtr *)cObject)->get();
    int tag = tensor->getSerializationTag();

    if(tag == features::internal::getIndexNumType<float>() + SERIALIZATION_HOMOGEN_TENSOR_ID ||
       tag == features::internal::getIndexNumType<float>() + SERIALIZATION_MKL_TENSOR_ID)
    {
        return (jint)(features::internal::getIndexNumType<float>());
    }
    if(tag == features::internal::getIndexNumType<double>() + SERIALIZATION_HOMOGEN_TENSOR_ID ||
       tag == features::internal::getIndexNumType<double>() + SERIALIZATION_MKL_TENSOR_ID)
    {
        return (jint)(features::internal::getIndexNumType<double>());
    }
    if(tag == features::internal::getIndexNumType<int>() + SERIALIZATION_HOMOGEN_TENSOR_ID)
    {
        return (jint)(features::internal::getIndexNumType<int>());
    }
    if(tag == features::internal::getIndexNumType<long>() + SERIALIZATION_HOMOGEN_TENSOR_ID)
    {
        return (jint)(features::internal::getIndexNumType<long>());
    }
    if(tag == features::internal::getIndexNumType<size_t>() + SERIALIZATION_HOMOGEN_TENSOR_ID)
    {
        return (jint)(features::internal::getIndexNumType<size_t>());
    }

    return (jint)features::DAAL_OTHER_T;
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenTensorByteBufferImpl
 * Method:    getDoubleSubtensorBuffer
 * Signature: (J[JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_HomogenTensorByteBufferImpl_getDoubleSubtensorBuffer
  (JNIEnv *env, jobject thisObject, jlong cObject, jlongArray jDims, jlong jRangeIdx, jlong jRangeNum, jobject jBuff)
{
    Tensor *tensor = static_cast<Tensor *>(((SerializationIfacePtr *)cObject)->get());
    SubtensorDescriptor<double> block;

    jsize len   = env->GetArrayLength(jDims);
    jlong *dimSizes = env->GetLongArrayElements(jDims, 0);
    Collection<size_t> dims;
    for(size_t i=0; i<len; i++)
    {
        dims.push_back( dimSizes[i] );
    }
    env->ReleaseLongArrayElements(jDims, dimSizes, 0);

    DAAL_CHECK_THROW(tensor->getSubtensor(dims.size(), &(dims[0]), (size_t)jRangeIdx, (size_t)jRangeNum, readOnly, block));
    double *data = block.getPtr();
    size_t  size = block.getSize();

    double *dst = (double *)(env->GetDirectBufferAddress(jBuff));

    for(size_t i = 0; i < size; i++)
    {
        dst[i] = data[i];
    }

    if(tensor->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tensor->getErrors()->getDescription());
    }

    DAAL_CHECK_THROW(tensor->releaseSubtensor(block));
    return jBuff;
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenTensorByteBufferImpl
 * Method:    getFloatSubtensorBuffer
 * Signature: (J[JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_HomogenTensorByteBufferImpl_getFloatSubtensorBuffer
  (JNIEnv *env, jobject thisObject, jlong cObject, jlongArray jDims, jlong jRangeIdx, jlong jRangeNum, jobject jBuff)
{
    Tensor *tensor = static_cast<Tensor *>(((SerializationIfacePtr *)cObject)->get());
    SubtensorDescriptor<float> block;

    jsize len   = env->GetArrayLength(jDims);
    jlong *dimSizes = env->GetLongArrayElements(jDims, 0);
    Collection<size_t> dims;
    for(size_t i=0; i<len; i++)
    {
        dims.push_back( dimSizes[i] );
    }
    env->ReleaseLongArrayElements(jDims, dimSizes, 0);

    DAAL_CHECK_THROW(tensor->getSubtensor(dims.size(), &(dims[0]), (size_t)jRangeIdx, (size_t)jRangeNum, readOnly, block));
    float *data = block.getPtr();
    size_t  size = block.getSize();

    float *dst = (float *)(env->GetDirectBufferAddress(jBuff));

    for(size_t i = 0; i < size; i++)
    {
        dst[i] = data[i];
    }

    if(tensor->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tensor->getErrors()->getDescription());
    }

    DAAL_CHECK_THROW(tensor->releaseSubtensor(block));
    return jBuff;
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenTensorByteBufferImpl
 * Method:    getIntSubtensorBuffer
 * Signature: (J[JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_HomogenTensorByteBufferImpl_getIntSubtensorBuffer
  (JNIEnv *env, jobject thisObject, jlong cObject, jlongArray jDims, jlong jRangeIdx, jlong jRangeNum, jobject jBuff)
{
    Tensor *tensor = static_cast<Tensor *>(((SerializationIfacePtr *)cObject)->get());
    SubtensorDescriptor<int> block;

    jsize len   = env->GetArrayLength(jDims);
    jlong *dimSizes = env->GetLongArrayElements(jDims, 0);
    Collection<size_t> dims;
    for(size_t i=0; i<len; i++)
    {
        dims.push_back( dimSizes[i] );
    }
    env->ReleaseLongArrayElements(jDims, dimSizes, 0);

    DAAL_CHECK_THROW(tensor->getSubtensor(dims.size(), &(dims[0]), (size_t)jRangeIdx, (size_t)jRangeNum, readOnly, block));
    int *data = block.getPtr();
    size_t  size = block.getSize();

    int *dst = (int *)(env->GetDirectBufferAddress(jBuff));

    for(size_t i = 0; i < size; i++)
    {
        dst[i] = data[i];
    }

    if(tensor->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tensor->getErrors()->getDescription());
    }

    DAAL_CHECK_THROW(tensor->releaseSubtensor(block));
    return jBuff;
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenTensorByteBufferImpl
 * Method:    releaseDoubleSubtensorBuffer
 * Signature: (J[JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenTensorByteBufferImpl_releaseDoubleSubtensorBuffer
  (JNIEnv *env, jobject thisObject, jlong cObject, jlongArray jDims, jlong jRangeIdx, jlong jRangeNum, jobject jBuff)
{
    Tensor *tensor = static_cast<Tensor *>(((SerializationIfacePtr *)cObject)->get());
    SubtensorDescriptor<double> block;

    jsize len   = env->GetArrayLength(jDims);
    jlong *dimSizes = env->GetLongArrayElements(jDims, 0);
    Collection<size_t> dims;
    for(size_t i=0; i<len; i++)
    {
        dims.push_back( dimSizes[i] );
    }
    env->ReleaseLongArrayElements(jDims, dimSizes, 0);

    DAAL_CHECK_THROW(tensor->getSubtensor(dims.size(), &(dims[0]), (size_t)jRangeIdx, (size_t)jRangeNum, writeOnly, block));
    double *data = block.getPtr();
    size_t  size = block.getSize();

    double *dst = (double *)(env->GetDirectBufferAddress(jBuff));

    for(size_t i = 0; i < size; i++)
    {
        data[i] = dst[i];
    }

    if(tensor->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tensor->getErrors()->getDescription());
    }

    DAAL_CHECK_THROW(tensor->releaseSubtensor(block));
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenTensorByteBufferImpl
 * Method:    releaseFloatSubtensorBuffer
 * Signature: (J[JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenTensorByteBufferImpl_releaseFloatSubtensorBuffer
  (JNIEnv *env, jobject thisObject, jlong cObject, jlongArray jDims, jlong jRangeIdx, jlong jRangeNum, jobject jBuff)
{
    Tensor *tensor = static_cast<Tensor *>(((SerializationIfacePtr *)cObject)->get());
    SubtensorDescriptor<float> block;

    jsize len   = env->GetArrayLength(jDims);
    jlong *dimSizes = env->GetLongArrayElements(jDims, 0);
    Collection<size_t> dims;
    for(size_t i=0; i<len; i++)
    {
        dims.push_back( dimSizes[i] );
    }
    env->ReleaseLongArrayElements(jDims, dimSizes, 0);

    DAAL_CHECK_THROW(tensor->getSubtensor(dims.size(), &(dims[0]), (size_t)jRangeIdx, (size_t)jRangeNum, writeOnly, block));
    float *data = block.getPtr();
    size_t  size = block.getSize();

    float *dst = (float *)(env->GetDirectBufferAddress(jBuff));

    for(size_t i = 0; i < size; i++)
    {
        data[i] = dst[i];
    }

    if(tensor->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tensor->getErrors()->getDescription());
    }

    DAAL_CHECK_THROW(tensor->releaseSubtensor(block));
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenTensorByteBufferImpl
 * Method:    releaseIntSubtensorBuffer
 * Signature: (J[JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenTensorByteBufferImpl_releaseIntSubtensorBuffer
  (JNIEnv *env, jobject thisObject, jlong cObject, jlongArray jDims, jlong jRangeIdx, jlong jRangeNum, jobject jBuff)
{
    Tensor *tensor = static_cast<Tensor *>(((SerializationIfacePtr *)cObject)->get());
    SubtensorDescriptor<int> block;

    jsize len   = env->GetArrayLength(jDims);
    jlong *dimSizes = env->GetLongArrayElements(jDims, 0);
    Collection<size_t> dims;
    for(size_t i=0; i<len; i++)
    {
        dims.push_back( dimSizes[i] );
    }
    env->ReleaseLongArrayElements(jDims, dimSizes, 0);

    DAAL_CHECK_THROW(tensor->getSubtensor(dims.size(), &(dims[0]), (size_t)jRangeIdx, (size_t)jRangeNum, writeOnly, block));
    int *data = block.getPtr();
    size_t  size = block.getSize();

    int *dst = (int *)(env->GetDirectBufferAddress(jBuff));

    for(size_t i = 0; i < size; i++)
    {
        data[i] = dst[i];
    }

    if(tensor->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tensor->getErrors()->getDescription());
    }

    DAAL_CHECK_THROW(tensor->releaseSubtensor(block));
}
