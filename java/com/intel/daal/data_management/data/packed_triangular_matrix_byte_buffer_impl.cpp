/* file: packed_triangular_matrix_byte_buffer_impl.cpp */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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

#include "com_intel_daal_data_management_data_PackedTriangularMatrixByteBufferImpl.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/symmetric_matrix.h"
#include "com/intel/daal/common_helpers_functions.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    getIndexType
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_getIndexType(JNIEnv * env, jobject thisobj,
                                                                                                                   jlong numTableAddr)
{
    NumericTable * nt             = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    NumericTableDictionary * dict = nt->getDictionary();
    return (jint)((*dict)[0].indexType);
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    dInit
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_dInit(JNIEnv * env, jobject thisobj,
                                                                                                             jlong nDim, jint layout)
{
    NumericTablePtr tbl;
    services::Status s;
    if (layout == NumericTableIface::StorageLayout::upperPackedTriangularMatrix)
    {
        tbl = PackedTriangularMatrix<NumericTableIface::StorageLayout::upperPackedTriangularMatrix, double>::create(
            nDim, NumericTableIface::AllocationFlag::doNotAllocate, &s);
    }
    else
    {
        tbl = PackedTriangularMatrix<NumericTableIface::StorageLayout::lowerPackedTriangularMatrix, double>::create(
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
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    sInit
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_sInit(JNIEnv * env, jobject thisobj,
                                                                                                             jlong nDim, jint layout)
{
    NumericTablePtr tbl;
    services::Status s;
    if (layout == NumericTableIface::StorageLayout::upperPackedTriangularMatrix)
    {
        tbl = PackedTriangularMatrix<NumericTableIface::StorageLayout::upperPackedTriangularMatrix, float>::create(
            nDim, NumericTableIface::AllocationFlag::doNotAllocate, &s);
    }
    else
    {
        tbl = PackedTriangularMatrix<NumericTableIface::StorageLayout::lowerPackedTriangularMatrix, float>::create(
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
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    lInit
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_lInit(JNIEnv * env, jobject thisobj,
                                                                                                             jlong nDim, jint layout)
{
    NumericTablePtr tbl;
    services::Status s;
    if (layout == NumericTableIface::StorageLayout::upperPackedTriangularMatrix)
    {
        tbl = PackedTriangularMatrix<NumericTableIface::StorageLayout::upperPackedTriangularMatrix, long>::create(
            nDim, NumericTableIface::AllocationFlag::doNotAllocate, &s);
    }
    else
    {
        tbl = PackedTriangularMatrix<NumericTableIface::StorageLayout::lowerPackedTriangularMatrix, long>::create(
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
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    iInit
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_iInit(JNIEnv * env, jobject thisobj,
                                                                                                             jlong nDim, jint layout)
{
    NumericTablePtr tbl;
    services::Status s;
    if (layout == NumericTableIface::StorageLayout::upperPackedTriangularMatrix)
    {
        tbl = PackedTriangularMatrix<NumericTableIface::StorageLayout::upperPackedTriangularMatrix, int>::create(
            nDim, NumericTableIface::AllocationFlag::doNotAllocate, &s);
    }
    else
    {
        tbl = PackedTriangularMatrix<NumericTableIface::StorageLayout::lowerPackedTriangularMatrix, int>::create(
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
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    releaseFloatBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_releaseFloatBlockBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<float> block;

    const size_t nCols = nt->getNumberOfColumns();
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
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    releaseDoubleBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_releaseDoubleBlockBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<double> block;

    const size_t nCols = nt->getNumberOfColumns();
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
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    releaseIntBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_releaseIntBlockBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<int> block;

    const size_t nCols = nt->getNumberOfColumns();
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
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    getDoubleBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_getDoubleBlockBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<double> block;

    const size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfRows(vectorIndex, vectorNum, readOnly, block));

    const double * data = block.getBlockPtr();

    double * dst = (double *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum * nCols; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    getFloatBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_getFloatBlockBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<float> block;

    const size_t nCols = nt->getNumberOfColumns();
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
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    getIntBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_getIntBlockBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<int> block;

    const size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfRows(vectorIndex, vectorNum, readOnly, block));

    const int * data = block.getBlockPtr();

    int * dst = (int *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum * nCols; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    getDoubleColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_getDoubleColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<double> block;

    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, readOnly, block));

    const double * data = block.getBlockPtr();

    double * dst = (double *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    getFloatColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_getFloatColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<float> block;

    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, readOnly, block));

    const float * data = block.getBlockPtr();

    float * dst = (float *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    getIntColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_getIntColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<int> block;

    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, readOnly, block));

    const int * data = block.getBlockPtr();

    int * dst = (int *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    releaseFloatColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_releaseFloatColumnBuffer(
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
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    releaseDoubleColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_releaseDoubleColumnBuffer(
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
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    releaseIntColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_releaseIntColumnBuffer(
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
 * Class:     com_intel_daal_data_management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    assignLong
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_assignLong(JNIEnv * env, jobject,
                                                                                                                 jlong numTableAddr, jlong constValue)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    if (nt->getDataLayout() == NumericTableIface::StorageLayout::upperPackedTriangularMatrix)
    {
        PackedTriangularMatrix<NumericTableIface::StorageLayout::upperPackedTriangularMatrix, long long> * ntPacked =
            static_cast<PackedTriangularMatrix<NumericTableIface::StorageLayout::upperPackedTriangularMatrix, long long> *>(
                ((SerializationIfacePtr *)numTableAddr)->get());
        ntPacked->assign((long long)constValue);
    }
    else
    {
        PackedTriangularMatrix<NumericTableIface::StorageLayout::lowerPackedTriangularMatrix, long long> * ntPacked =
            static_cast<PackedTriangularMatrix<NumericTableIface::StorageLayout::lowerPackedTriangularMatrix, long long> *>(
                ((SerializationIfacePtr *)numTableAddr)->get());
        ntPacked->assign((long long)constValue);
    }
}

/*
 * Class:     com_intel_daal_data_management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    assignInt
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_assignInt(JNIEnv * env, jobject,
                                                                                                                jlong numTableAddr, jint constValue)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    if (nt->getDataLayout() == NumericTableIface::StorageLayout::upperPackedTriangularMatrix)
    {
        PackedTriangularMatrix<NumericTableIface::StorageLayout::upperPackedTriangularMatrix, int> * ntPacked =
            static_cast<PackedTriangularMatrix<NumericTableIface::StorageLayout::upperPackedTriangularMatrix, int> *>(
                ((SerializationIfacePtr *)numTableAddr)->get());
        ntPacked->assign((int)constValue);
    }
    else
    {
        PackedTriangularMatrix<NumericTableIface::StorageLayout::lowerPackedTriangularMatrix, int> * ntPacked =
            static_cast<PackedTriangularMatrix<NumericTableIface::StorageLayout::lowerPackedTriangularMatrix, int> *>(
                ((SerializationIfacePtr *)numTableAddr)->get());
        ntPacked->assign((int)constValue);
    }
}

/*
 * Class:     com_intel_daal_data_management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    assignDouble
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_assignDouble(JNIEnv * env, jobject,
                                                                                                                   jlong numTableAddr,
                                                                                                                   jdouble constValue)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    if (nt->getDataLayout() == NumericTableIface::StorageLayout::upperPackedTriangularMatrix)
    {
        PackedTriangularMatrix<NumericTableIface::StorageLayout::upperPackedTriangularMatrix, double> * ntPacked =
            static_cast<PackedTriangularMatrix<NumericTableIface::StorageLayout::upperPackedTriangularMatrix, double> *>(
                ((SerializationIfacePtr *)numTableAddr)->get());
        ntPacked->assign((double)constValue);
    }
    else
    {
        PackedTriangularMatrix<NumericTableIface::StorageLayout::lowerPackedTriangularMatrix, double> * ntPacked =
            static_cast<PackedTriangularMatrix<NumericTableIface::StorageLayout::lowerPackedTriangularMatrix, double> *>(
                ((SerializationIfacePtr *)numTableAddr)->get());
        ntPacked->assign((double)constValue);
    }
}

/*
 * Class:     com_intel_daal_data_management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    assignFloat
 * Signature: (JF)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_assignFloat(JNIEnv * env, jobject,
                                                                                                                  jlong numTableAddr,
                                                                                                                  jfloat constValue)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    if (nt->getDataLayout() == NumericTableIface::StorageLayout::upperPackedTriangularMatrix)
    {
        PackedTriangularMatrix<NumericTableIface::StorageLayout::upperPackedTriangularMatrix, float> * ntPacked =
            static_cast<PackedTriangularMatrix<NumericTableIface::StorageLayout::upperPackedTriangularMatrix, float> *>(
                ((SerializationIfacePtr *)numTableAddr)->get());
        ntPacked->assign((float)constValue);
    }
    else
    {
        PackedTriangularMatrix<NumericTableIface::StorageLayout::lowerPackedTriangularMatrix, float> * ntPacked =
            static_cast<PackedTriangularMatrix<NumericTableIface::StorageLayout::lowerPackedTriangularMatrix, float> *>(
                ((SerializationIfacePtr *)numTableAddr)->get());
        ntPacked->assign((float)constValue);
    }
}

/*
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    getDoublePackedBuffer
 * Signature:(JILjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_getDoublePackedBuffer(JNIEnv * env,
                                                                                                                               jobject thisObj,
                                                                                                                               jlong numTableAddr,
                                                                                                                               jobject byteBuffer)
{
    PackedArrayNumericTableIface * nt = dynamic_cast<PackedArrayNumericTableIface *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<double> block;

    DAAL_CHECK_THROW(nt->getPackedArray(readOnly, block));

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
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    getFloatPackedBuffer
 * Signature:(JILjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_getFloatPackedBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jint nDim, jobject byteBuffer)
{
    PackedArrayNumericTableIface * nt = dynamic_cast<PackedArrayNumericTableIface *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<float> block;

    DAAL_CHECK_THROW(nt->getPackedArray(readOnly, block));

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
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    getIntPackedBuffer
 * Signature:(JILjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_getIntPackedBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jint nDim, jobject byteBuffer)
{
    PackedArrayNumericTableIface * nt = dynamic_cast<PackedArrayNumericTableIface *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<int> block;

    DAAL_CHECK_THROW(nt->getPackedArray(readOnly, block));

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
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    releaseDoublePackedBuffer
 * Signature:(JLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_releaseDoublePackedBuffer(JNIEnv * env,
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
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    releaseFloatPackedBuffer
 * Signature:(JLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_releaseFloatPackedBuffer(JNIEnv * env,
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
 * Class:     com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    releaseIntPackedBuffer
 * Signature:(JLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_releaseIntPackedBuffer(JNIEnv * env,
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
 * Class:     com_intel_daal_data_management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    cAllocateDataMemoryDouble
 * Signature:(J)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_cAllocateDataMemoryDouble(JNIEnv * env,
                                                                                                                                jobject thisObj,
                                                                                                                                jlong numTableAddr)
{
    NumericTable * tbl = ((NumericTablePtr *)numTableAddr)->get();
    services::Status s;
    if (tbl->getDataLayout() == NumericTableIface::StorageLayout::upperPackedTriangularMatrix)
    {
        s = ((PackedTriangularMatrix<NumericTableIface::StorageLayout::upperPackedTriangularMatrix, double> *)tbl)->allocateDataMemory();
    }
    else
    {
        s = ((PackedTriangularMatrix<NumericTableIface::StorageLayout::lowerPackedTriangularMatrix, double> *)tbl)->allocateDataMemory();
    }
    DAAL_CHECK_THROW(s);
}

/*
 * Class:     com_intel_daal_data_management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    cAllocateDataMemoryFloat
 * Signature:(J)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_cAllocateDataMemoryFloat(JNIEnv * env,
                                                                                                                               jobject thisObj,
                                                                                                                               jlong numTableAddr)
{
    NumericTable * tbl = ((NumericTablePtr *)numTableAddr)->get();
    services::Status s;

    if (tbl->getDataLayout() == NumericTableIface::StorageLayout::upperPackedTriangularMatrix)
    {
        s = ((PackedTriangularMatrix<NumericTableIface::StorageLayout::upperPackedTriangularMatrix, float> *)tbl)->allocateDataMemory();
    }
    else
    {
        s = ((PackedTriangularMatrix<NumericTableIface::StorageLayout::lowerPackedTriangularMatrix, float> *)tbl)->allocateDataMemory();
    }

    DAAL_CHECK_THROW(s);
}

/*
 * Class:     com_intel_daal_data_management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    cAllocateDataMemoryLong
 * Signature:(J)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_cAllocateDataMemoryLong(JNIEnv * env,
                                                                                                                              jobject thisObj,
                                                                                                                              jlong numTableAddr)
{
    NumericTable * tbl = ((NumericTablePtr *)numTableAddr)->get();
    services::Status s;

    if (tbl->getDataLayout() == NumericTableIface::StorageLayout::upperPackedTriangularMatrix)
    {
        s = ((PackedTriangularMatrix<NumericTableIface::StorageLayout::upperPackedTriangularMatrix, long> *)tbl)->allocateDataMemory();
    }
    else
    {
        s = ((PackedTriangularMatrix<NumericTableIface::StorageLayout::lowerPackedTriangularMatrix, long> *)tbl)->allocateDataMemory();
    }

    DAAL_CHECK_THROW(s);
}

/*
 * Class:     com_intel_daal_data_management_data_PackedTriangularMatrixByteBufferImpl
 * Method:    cAllocateDataMemoryInt
 * Signature:(J)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_PackedTriangularMatrixByteBufferImpl_cAllocateDataMemoryInt(JNIEnv * env,
                                                                                                                             jobject thisObj,
                                                                                                                             jlong numTableAddr)
{
    NumericTable * tbl = ((NumericTablePtr *)numTableAddr)->get();

    services::Status s;
    if (tbl->getDataLayout() == NumericTableIface::StorageLayout::upperPackedTriangularMatrix)
    {
        s = ((PackedTriangularMatrix<NumericTableIface::StorageLayout::upperPackedTriangularMatrix, int> *)tbl)->allocateDataMemory();
    }
    else
    {
        s = ((PackedTriangularMatrix<NumericTableIface::StorageLayout::lowerPackedTriangularMatrix, int> *)tbl)->allocateDataMemory();
    }

    DAAL_CHECK_THROW(s);
}
