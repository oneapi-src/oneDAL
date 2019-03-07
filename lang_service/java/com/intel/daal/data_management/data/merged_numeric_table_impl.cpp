/* file: merged_numeric_table_impl.cpp */
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

#include "JMergedNumericTableImpl.h"
#include "numeric_table.h"
#include "merged_numeric_table.h"
#include "daal.h"
#include "common_helpers_functions.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_data_MergedNumericTableImpl
 * Method:    cNewMergedNumericTable
 * Signature:()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_MergedNumericTableImpl_cNewMergedNumericTable
(JNIEnv *env, jobject thisObj)
{
    // Create C++ object of the class NumericTable
    NumericTablePtr *tbl = new NumericTablePtr(new MergedNumericTable());

    /*if((*tbl)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), (*tbl)->getErrors()->getDescription());
    }*/

    return(jlong)tbl;
}

/*
 * Class:     com_intel_daal_data_1management_data_MergedNumericTableImpl
 * Method:    cAddDataCollection
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_MergedNumericTableImpl_cAddNumericTable
(JNIEnv *env, jobject thisObj, jlong mergedNumericTableAddr, jlong numericTableAddr)
{
    data_management::MergedNumericTablePtr pMergedNumericTable =
            (*(data_management::MergedNumericTablePtr *)mergedNumericTableAddr);
    data_management::NumericTablePtr pNumericTable =
            (*(data_management::NumericTablePtr *)numericTableAddr);
    pMergedNumericTable->addNumericTable(pNumericTable);
}

/*
 * Class:     com_intel_daal_data_1management_data_MergedNumericTableImpl
 * Method:    cGetNumberOfColumns
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_MergedNumericTableImpl_cGetNumberOfColumns
(JNIEnv *env, jobject thisObj, jlong mergedNumericTableAddr)
{
    data_management::MergedNumericTablePtr pMergedNumericTable =
            (*(data_management::MergedNumericTablePtr *)mergedNumericTableAddr);
    return pMergedNumericTable->getNumberOfColumns();
}

/*
 * Class:     com_intel_daal_data_1management_data_MergedNumericTableImpl
 * Method:    releaseFloatBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_MergedNumericTableImpl_releaseFloatBlockBuffer
(JNIEnv *env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    MergedNumericTable *nt = (*(MergedNumericTablePtr *)numTableAddr).get();
    BlockDescriptor<float> block;

    const size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfRows(vectorIndex, vectorNum, writeOnly, block));

    float* data = block.getBlockPtr();
    const float *src = (float *)(env->GetDirectBufferAddress(byteBuffer));

    for(size_t i = 0; i < vectorNum * nCols; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
}

/*
 * Class:     com_intel_daal_data_1management_data_MergedNumericTableImpl
 * Method:    releaseDoubleBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_MergedNumericTableImpl_releaseDoubleBlockBuffer
(JNIEnv *env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    MergedNumericTable *nt = (*(MergedNumericTablePtr *)numTableAddr).get();
    BlockDescriptor<double> block;

    const size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfRows(vectorIndex, vectorNum, writeOnly, block));

    double *data = block.getBlockPtr();

    const double *src = (double *)(env->GetDirectBufferAddress(byteBuffer));

    for(size_t i = 0; i < vectorNum * nCols; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
}

/*
 * Class:     com_intel_daal_data_1management_data_MergedNumericTableImpl
 * Method:    releaseIntBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_MergedNumericTableImpl_releaseIntBlockBuffer
(JNIEnv *env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    MergedNumericTable *nt = (*(MergedNumericTablePtr *)numTableAddr).get();
    BlockDescriptor<int> block;

    const size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfRows(vectorIndex, vectorNum, writeOnly, block));

    int* data = block.getBlockPtr();
    const int *src = (int *)(env->GetDirectBufferAddress(byteBuffer));

    for(size_t i = 0; i < vectorNum * nCols; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
}

/*
 * Class:     com_intel_daal_data_1management_data_MergedNumericTableImpl
 * Method:    getDoubleBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_MergedNumericTableImpl_getDoubleBlockBuffer
(JNIEnv *env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    MergedNumericTable *nt = (*(MergedNumericTablePtr *)numTableAddr).get();
    BlockDescriptor<double> block;

    const size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfRows(vectorIndex, vectorNum, readOnly, block));

    const double *data = block.getBlockPtr();

    double *dst = (double *)(env->GetDirectBufferAddress(byteBuffer));

    for(size_t i = 0; i < vectorNum * nCols; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_MergedNumericTableImpl
 * Method:    getFloatBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_MergedNumericTableImpl_getFloatBlockBuffer
(JNIEnv *env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    MergedNumericTable *nt = (*(MergedNumericTablePtr *)numTableAddr).get();
    BlockDescriptor<float> block;

    const size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfRows(vectorIndex, vectorNum, readOnly, block));

    const float *data = block.getBlockPtr();

    float *dst = (float *)(env->GetDirectBufferAddress(byteBuffer));

    for(size_t i = 0; i < vectorNum * nCols; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_MergedNumericTableImpl
 * Method:    getIntBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_MergedNumericTableImpl_getIntBlockBuffer
(JNIEnv *env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    MergedNumericTable *nt = (*(MergedNumericTablePtr *)numTableAddr).get();
    BlockDescriptor<int> block;

    const size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfRows(vectorIndex, vectorNum, readOnly, block));

    const int *data = block.getBlockPtr();

    int *dst = (int *)(env->GetDirectBufferAddress(byteBuffer));

    for(size_t i = 0; i < vectorNum * nCols; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));

    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_MergedNumericTableImpl
 * Method:    getDoubleColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_MergedNumericTableImpl_getDoubleColumnBuffer
(JNIEnv *env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    MergedNumericTable *nt = (*(MergedNumericTablePtr *)numTableAddr).get();
    BlockDescriptor<double> block;

    size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, readOnly, block));

    const double *data = block.getBlockPtr();

    double *dst = (double *)(env->GetDirectBufferAddress(byteBuffer));

    for(size_t i = 0; i < vectorNum; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_MergedNumericTableImpl
 * Method:    getFloatColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_MergedNumericTableImpl_getFloatColumnBuffer
(JNIEnv *env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    MergedNumericTable *nt = (*(MergedNumericTablePtr *)numTableAddr).get();
    BlockDescriptor<float> block;

    const size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, readOnly, block));

    const float *data = block.getBlockPtr();

    float *dst = (float *)(env->GetDirectBufferAddress(byteBuffer));

    for(size_t i = 0; i < vectorNum; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_MergedNumericTableImpl
 * Method:    getIntColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_MergedNumericTableImpl_getIntColumnBuffer
(JNIEnv *env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    MergedNumericTable *nt = (*(MergedNumericTablePtr *)numTableAddr).get();
    BlockDescriptor<int> block;

    const size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, readOnly, block));

    const int *data = block.getBlockPtr();

    int *dst = (int *)(env->GetDirectBufferAddress(byteBuffer));

    for(size_t i = 0; i < vectorNum; i++)
    {
        dst[i] = data[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_MergedNumericTableImpl
 * Method:    releaseFloatColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_MergedNumericTableImpl_releaseFloatColumnBuffer
(JNIEnv *env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    MergedNumericTable *nt = (*(MergedNumericTablePtr *)numTableAddr).get();
    BlockDescriptor<float> block;

    const size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, writeOnly, block));

    float* data = block.getBlockPtr();

    const float *src = (float *)(env->GetDirectBufferAddress(byteBuffer));

    for(size_t i = 0; i < vectorNum; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
}

/*
 * Class:     com_intel_daal_data_1management_data_MergedNumericTableImpl
 * Method:    releaseDoubleColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_MergedNumericTableImpl_releaseDoubleColumnBuffer
(JNIEnv *env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    MergedNumericTable *nt = (*(MergedNumericTablePtr *)numTableAddr).get();
    BlockDescriptor<double> block;

    const size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, writeOnly, block));

    double *data = block.getBlockPtr();

    const double *src = (double *)(env->GetDirectBufferAddress(byteBuffer));

    for(size_t i = 0; i < vectorNum; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
}

/*
 * Class:     com_intel_daal_data_1management_data_MergedNumericTableImpl
 * Method:    releaseIntColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_MergedNumericTableImpl_releaseIntColumnBuffer
(JNIEnv *env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    MergedNumericTable *nt = (*(MergedNumericTablePtr *)numTableAddr).get();
    BlockDescriptor<int> block;

    size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, writeOnly, block));

    int* data = block.getBlockPtr();

    const int *src = (int *)(env->GetDirectBufferAddress(byteBuffer));

    for(size_t i = 0; i < vectorNum; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
}
