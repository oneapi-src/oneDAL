/* file: packed_symmetric_matrix_byte_buffer_impl.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "com_intel_daal_data_management_data_PackedSymmetricMatrixByteBufferImpl.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/symmetric_matrix.h"
#include "com/intel/daal/common_helpers_functions.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    getIndexType
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_getIndexType(JNIEnv * env, jobject thisobj,
                                                                                                                  jlong numTableAddr)
{
    NumericTable * nt             = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    NumericTableDictionary * dict = nt->getDictionary();
    return (jint)((*dict)[0].indexType);
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    dInit
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_dInit(JNIEnv * env, jobject thisobj, jlong nDim,
                                                                                                            jint layout)
{
    NumericTablePtr tbl;
    services::Status s;
    if (layout == NumericTableIface::StorageLayout::upperPackedSymmetricMatrix)
    {
        tbl = PackedSymmetricMatrix<NumericTableIface::StorageLayout::upperPackedSymmetricMatrix, double>::create(
            nDim, NumericTableIface::AllocationFlag::doNotAllocate, &s);
    }
    else
    {
        tbl = PackedSymmetricMatrix<NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix, double>::create(
            nDim, NumericTableIface::AllocationFlag::doNotAllocate, &s);
    }
    if (!s)
    {
        tbl = NumericTablePtr();
        DAAL_CHECK_THROW(s);
    }
    return (jlong) new SerializationIfacePtr(tbl);
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    sInit
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_sInit(JNIEnv * env, jobject thisobj, jlong nDim,
                                                                                                            jint layout)
{
    NumericTablePtr tbl;
    services::Status s;
    if (layout == NumericTableIface::StorageLayout::upperPackedSymmetricMatrix)
    {
        tbl = PackedSymmetricMatrix<NumericTableIface::StorageLayout::upperPackedSymmetricMatrix, float>::create(
            nDim, NumericTableIface::AllocationFlag::doNotAllocate, &s);
    }
    else
    {
        tbl = PackedSymmetricMatrix<NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix, float>::create(
            nDim, NumericTableIface::AllocationFlag::doNotAllocate, &s);
    }
    if (!s)
    {
        tbl = NumericTablePtr();
        DAAL_CHECK_THROW(s);
    }
    return (jlong) new SerializationIfacePtr(tbl);
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    lInit
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_lInit(JNIEnv * env, jobject thisobj, jlong nDim,
                                                                                                            jint layout)
{
    NumericTablePtr tbl;
    services::Status s;
    if (layout == NumericTableIface::StorageLayout::upperPackedSymmetricMatrix)
    {
        tbl = PackedSymmetricMatrix<NumericTableIface::StorageLayout::upperPackedSymmetricMatrix, long>::create(
            nDim, NumericTableIface::AllocationFlag::doNotAllocate, &s);
    }
    else
    {
        tbl = PackedSymmetricMatrix<NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix, long>::create(
            nDim, NumericTableIface::AllocationFlag::doNotAllocate, &s);
    }
    if (!s)
    {
        tbl = NumericTablePtr();
        DAAL_CHECK_THROW(s);
    }
    return (jlong) new SerializationIfacePtr(tbl);
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    iInit
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_iInit(JNIEnv * env, jobject thisobj, jlong nDim,
                                                                                                            jint layout)
{
    NumericTablePtr tbl;
    services::Status s;
    if (layout == NumericTableIface::StorageLayout::upperPackedSymmetricMatrix)
    {
        tbl = PackedSymmetricMatrix<NumericTableIface::StorageLayout::upperPackedSymmetricMatrix, int>::create(
            nDim, NumericTableIface::AllocationFlag::doNotAllocate, &s);
    }
    else
    {
        tbl = PackedSymmetricMatrix<NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix, int>::create(
            nDim, NumericTableIface::AllocationFlag::doNotAllocate, &s);
    }
    if (!s)
    {
        tbl = NumericTablePtr();
        DAAL_CHECK_THROW(s);
    }
    return (jlong) new SerializationIfacePtr(tbl);
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    releaseFloatBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_releaseFloatBlockBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<float> block;

    size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfRows(vectorIndex, vectorNum, writeOnly, block));

    float * data      = block.getBlockPtr();
    const float * src = (float *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum * nCols; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    releaseDoubleBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_releaseDoubleBlockBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<double> block;

    size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfRows(vectorIndex, vectorNum, writeOnly, block));

    double * data      = block.getBlockPtr();
    const double * src = (double *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum * nCols; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    releaseIntBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_releaseIntBlockBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<int> block;

    size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfRows(vectorIndex, vectorNum, writeOnly, block));

    int * data      = block.getBlockPtr();
    const int * src = (int *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum * nCols; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    getDoubleBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_getDoubleBlockBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<double> block;

    size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfRows(vectorIndex, vectorNum, readOnly, block));

    const double * data = block.getBlockPtr();
    double * dst        = (double *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum * nCols; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    getFloatBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_getFloatBlockBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<float> block;

    size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfRows(vectorIndex, vectorNum, readOnly, block));

    const float * data = block.getBlockPtr();

    float * dst = (float *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum * nCols; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    getIntBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_getIntBlockBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<int> block;

    size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfRows(vectorIndex, vectorNum, readOnly, block));

    const int * data = block.getBlockPtr();
    int * dst        = (int *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum * nCols; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    getDoubleColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_getDoubleColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<double> block;

    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, readOnly, block));

    const double * data = block.getBlockPtr();
    double * dst        = (double *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    getFloatColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_getFloatColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<float> block;

    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, readOnly, block));

    const float * data = block.getBlockPtr();
    float * dst        = (float *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    getIntColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_getIntColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<int> block;

    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, readOnly, block));

    const int * data = block.getBlockPtr();
    int * dst        = (int *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    releaseFloatColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_releaseFloatColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<float> block;

    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, writeOnly, block));

    float * data = block.getBlockPtr();

    const float * src = (float *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    releaseDoubleColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_releaseDoubleColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<double> block;

    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, writeOnly, block));

    double * data = block.getBlockPtr();

    const double * src = (double *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    releaseIntColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_releaseIntColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<int> block;

    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, writeOnly, block));

    int * data = block.getBlockPtr();

    const int * src = (int *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
}

/*
 * Class:     com_intel_daal_data_management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    assignLong
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_assignLong(JNIEnv * env, jobject,
                                                                                                                jlong numTableAddr, jlong constValue)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    if (nt->getDataLayout() == NumericTableIface::StorageLayout::upperPackedSymmetricMatrix)
    {
        PackedSymmetricMatrix<NumericTableIface::StorageLayout::upperPackedSymmetricMatrix, long long> * ntPacked =
            static_cast<PackedSymmetricMatrix<NumericTableIface::StorageLayout::upperPackedSymmetricMatrix, long long> *>(
                ((SerializationIfacePtr *)numTableAddr)->get());
        ntPacked->assign((long long)constValue);
    }
    else
    {
        PackedSymmetricMatrix<NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix, long long> * ntPacked =
            static_cast<PackedSymmetricMatrix<NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix, long long> *>(
                ((SerializationIfacePtr *)numTableAddr)->get());
        ntPacked->assign((long long)constValue);
    }
}

/*
 * Class:     com_intel_daal_data_management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    assignInt
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_assignInt(JNIEnv * env, jobject,
                                                                                                               jlong numTableAddr, jint constValue)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    if (nt->getDataLayout() == NumericTableIface::StorageLayout::upperPackedSymmetricMatrix)
    {
        PackedSymmetricMatrix<NumericTableIface::StorageLayout::upperPackedSymmetricMatrix, int> * ntPacked =
            static_cast<PackedSymmetricMatrix<NumericTableIface::StorageLayout::upperPackedSymmetricMatrix, int> *>(
                ((SerializationIfacePtr *)numTableAddr)->get());
        ntPacked->assign((int)constValue);
    }
    else
    {
        PackedSymmetricMatrix<NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix, int> * ntPacked =
            static_cast<PackedSymmetricMatrix<NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix, int> *>(
                ((SerializationIfacePtr *)numTableAddr)->get());
        ntPacked->assign((int)constValue);
    }
}

/*
 * Class:     com_intel_daal_data_management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    assignDouble
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_assignDouble(JNIEnv * env, jobject,
                                                                                                                  jlong numTableAddr,
                                                                                                                  jdouble constValue)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    if (nt->getDataLayout() == NumericTableIface::StorageLayout::upperPackedSymmetricMatrix)
    {
        PackedSymmetricMatrix<NumericTableIface::StorageLayout::upperPackedSymmetricMatrix, double> * ntPacked =
            static_cast<PackedSymmetricMatrix<NumericTableIface::StorageLayout::upperPackedSymmetricMatrix, double> *>(
                ((SerializationIfacePtr *)numTableAddr)->get());
        ntPacked->assign((double)constValue);
    }
    else
    {
        PackedSymmetricMatrix<NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix, double> * ntPacked =
            static_cast<PackedSymmetricMatrix<NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix, double> *>(
                ((SerializationIfacePtr *)numTableAddr)->get());
        ntPacked->assign((double)constValue);
    }
}

/*
 * Class:     com_intel_daal_data_management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    assignFloat
 * Signature: (JF)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_assignFloat(JNIEnv * env, jobject,
                                                                                                                 jlong numTableAddr,
                                                                                                                 jfloat constValue)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    if (nt->getDataLayout() == NumericTableIface::StorageLayout::upperPackedSymmetricMatrix)
    {
        PackedSymmetricMatrix<NumericTableIface::StorageLayout::upperPackedSymmetricMatrix, float> * ntPacked =
            static_cast<PackedSymmetricMatrix<NumericTableIface::StorageLayout::upperPackedSymmetricMatrix, float> *>(
                ((SerializationIfacePtr *)numTableAddr)->get());
        ntPacked->assign((float)constValue);
    }
    else
    {
        PackedSymmetricMatrix<NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix, float> * ntPacked =
            static_cast<PackedSymmetricMatrix<NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix, float> *>(
                ((SerializationIfacePtr *)numTableAddr)->get());
        ntPacked->assign((float)constValue);
    }
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    getDoublePackedBuffer
 * Signature:(JILjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_getDoublePackedBuffer(JNIEnv * env,
                                                                                                                              jobject thisObj,
                                                                                                                              jlong numTableAddr,
                                                                                                                              jobject byteBuffer)
{
    PackedArrayNumericTableIface * nt = dynamic_cast<PackedArrayNumericTableIface *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<double> block;

    nt->getPackedArray(readOnly, block);

    double * data = block.getBlockPtr();
    size_t nSize  = block.getNumberOfColumns();

    double * dst = (double *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < nSize; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releasePackedArray(block));

    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    getFloatPackedBuffer
 * Signature:(JILjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_getFloatPackedBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jint nDim, jobject byteBuffer)
{
    PackedArrayNumericTableIface * nt = dynamic_cast<PackedArrayNumericTableIface *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<float> block;

    nt->getPackedArray(readOnly, block);

    float * data = block.getBlockPtr();
    size_t nSize = block.getNumberOfColumns();

    float * dst = (float *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < nSize; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releasePackedArray(block));

    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    getIntPackedBuffer
 * Signature:(JILjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_getIntPackedBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jint nDim, jobject byteBuffer)
{
    PackedArrayNumericTableIface * nt = dynamic_cast<PackedArrayNumericTableIface *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<int> block;

    nt->getPackedArray(readOnly, block);

    int * data   = block.getBlockPtr();
    size_t nSize = block.getNumberOfColumns();

    int * dst = (int *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < nSize; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releasePackedArray(block));

    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    releaseDoublePackedBuffer
 * Signature:(JLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_releaseDoublePackedBuffer(JNIEnv * env,
                                                                                                                                  jobject thisObj,
                                                                                                                                  jlong numTableAddr,
                                                                                                                                  jobject byteBuffer)
{
    PackedArrayNumericTableIface * nt = dynamic_cast<PackedArrayNumericTableIface *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<double> block;

    DAAL_CHECK_THROW(nt->getPackedArray(writeOnly, block));

    double * data = block.getBlockPtr();
    size_t nSize  = block.getNumberOfColumns();

    double * src = (double *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < nSize; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releasePackedArray(block));

    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    releaseFloatPackedBuffer
 * Signature:(JLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_releaseFloatPackedBuffer(JNIEnv * env,
                                                                                                                                 jobject thisObj,
                                                                                                                                 jlong numTableAddr,
                                                                                                                                 jobject byteBuffer)
{
    PackedArrayNumericTableIface * nt = dynamic_cast<PackedArrayNumericTableIface *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<float> block;

    DAAL_CHECK_THROW(nt->getPackedArray(writeOnly, block));

    float * data = block.getBlockPtr();
    size_t nSize = block.getNumberOfColumns();

    float * src = (float *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < nSize; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releasePackedArray(block));

    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    releaseIntPackedBuffer
 * Signature:(JLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_releaseIntPackedBuffer(JNIEnv * env,
                                                                                                                               jobject thisObj,
                                                                                                                               jlong numTableAddr,
                                                                                                                               jobject byteBuffer)
{
    PackedArrayNumericTableIface * nt = dynamic_cast<PackedArrayNumericTableIface *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<int> block;

    DAAL_CHECK_THROW(nt->getPackedArray(writeOnly, block));

    int * data   = block.getBlockPtr();
    size_t nSize = block.getNumberOfColumns();

    int * src = (int *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < nSize; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releasePackedArray(block));

    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    cAllocateDataMemoryDouble
 * Signature:(J)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_cAllocateDataMemoryDouble(JNIEnv * env,
                                                                                                                               jobject thisObj,
                                                                                                                               jlong numTableAddr)
{
    NumericTable * tbl = ((NumericTablePtr *)numTableAddr)->get();

    services::Status s;
    if (tbl->getDataLayout() == NumericTableIface::StorageLayout::upperPackedSymmetricMatrix)
    {
        s = ((PackedSymmetricMatrix<NumericTableIface::StorageLayout::upperPackedSymmetricMatrix, double> *)tbl)->allocateDataMemory();
    }
    else
    {
        s = ((PackedSymmetricMatrix<NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix, double> *)tbl)->allocateDataMemory();
    }
    DAAL_CHECK_THROW(s);
}

/*
 * Class:     com_intel_daal_data_management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    cAllocateDataMemoryFloat
 * Signature:(J)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_cAllocateDataMemoryFloat(JNIEnv * env,
                                                                                                                              jobject thisObj,
                                                                                                                              jlong numTableAddr)
{
    NumericTable * tbl = ((NumericTablePtr *)numTableAddr)->get();

    services::Status s;
    if (tbl->getDataLayout() == NumericTableIface::StorageLayout::upperPackedSymmetricMatrix)
    {
        s = ((PackedSymmetricMatrix<NumericTableIface::StorageLayout::upperPackedSymmetricMatrix, float> *)tbl)->allocateDataMemory();
    }
    else
    {
        s = ((PackedSymmetricMatrix<NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix, float> *)tbl)->allocateDataMemory();
    }
    DAAL_CHECK_THROW(s);
}

/*
 * Class:     com_intel_daal_data_management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    cAllocateDataMemoryLong
 * Signature:(J)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_cAllocateDataMemoryLong(JNIEnv * env,
                                                                                                                             jobject thisObj,
                                                                                                                             jlong numTableAddr)
{
    NumericTable * tbl = ((NumericTablePtr *)numTableAddr)->get();

    services::Status s;
    if (tbl->getDataLayout() == NumericTableIface::StorageLayout::upperPackedSymmetricMatrix)
    {
        s = ((PackedSymmetricMatrix<NumericTableIface::StorageLayout::upperPackedSymmetricMatrix, long> *)tbl)->allocateDataMemory();
    }
    else
    {
        s = ((PackedSymmetricMatrix<NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix, long> *)tbl)->allocateDataMemory();
    }
    DAAL_CHECK_THROW(s);
}

/*
 * Class:     com_intel_daal_data_management_data_PackedSymmetricMatrixByteBufferImpl
 * Method:    cAllocateDataMemoryInt
 * Signature:(J)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedSymmetricMatrixByteBufferImpl_cAllocateDataMemoryInt(JNIEnv * env,
                                                                                                                            jobject thisObj,
                                                                                                                            jlong numTableAddr)
{
    NumericTable * tbl = ((NumericTablePtr *)numTableAddr)->get();

    services::Status s;
    if (tbl->getDataLayout() == NumericTableIface::StorageLayout::upperPackedSymmetricMatrix)
    {
        s = ((PackedSymmetricMatrix<NumericTableIface::StorageLayout::upperPackedSymmetricMatrix, int> *)tbl)->allocateDataMemory();
    }
    else
    {
        s = ((PackedSymmetricMatrix<NumericTableIface::StorageLayout::lowerPackedSymmetricMatrix, int> *)tbl)->allocateDataMemory();
    }
    DAAL_CHECK_THROW(s);
}
