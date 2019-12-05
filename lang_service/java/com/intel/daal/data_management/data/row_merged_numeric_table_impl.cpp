/* file: row_merged_numeric_table_impl.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include "com_intel_daal_data_management_data_RowMergedNumericTableImpl.h"
#include "numeric_table.h"
#include "row_merged_numeric_table.h"
#include "daal.h"
#include "common_helpers_functions.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_data_RowMergedNumericTableImpl
 * Method:    cNewRowMergedNumericTable
 * Signature:()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_cNewRowMergedNumericTable(JNIEnv * env, jobject thisObj)
{
    // Create C++ object of the class NumericTable
    NumericTable * tbl = new RowMergedNumericTable();
    /*if(!tbl->status())
    {
        services::Status s = tbl->status();
        delete tbl;
        DAAL_CHECK_THROW(s);
    }*/
    return (jlong) new NumericTablePtr(tbl);
}

/*
 * Class:     com_intel_daal_data_1management_data_RowMergedNumericTableImpl
 * Method:    cAddDataCollection
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_cAddNumericTable(JNIEnv * env, jobject thisObj,
                                                                                                            jlong rowMergedNumericTableAddr,
                                                                                                            jlong numericTableAddr)
{
    data_management::RowMergedNumericTablePtr pRowMergedNumericTable = (*(data_management::RowMergedNumericTablePtr *)rowMergedNumericTableAddr);
    data_management::NumericTablePtr pNumericTable                   = (*(data_management::NumericTablePtr *)numericTableAddr);
    pRowMergedNumericTable->addNumericTable(pNumericTable);
}

/*
 * Class:     com_intel_daal_data_1management_data_RowMergedNumericTableImpl
 * Method:    cGetNumberOfColumns
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_cGetNumberOfColumns(JNIEnv * env, jobject thisObj,
                                                                                                                jlong rowMergedNumericTableAddr)
{
    data_management::RowMergedNumericTablePtr pRowMergedNumericTable = (*(data_management::RowMergedNumericTablePtr *)rowMergedNumericTableAddr);
    return pRowMergedNumericTable->getNumberOfColumns();
}

/*
 * Class:     com_intel_daal_data_1management_data_RowMergedNumericTableImpl
 * Method:    releaseFloatBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_releaseFloatBlockBuffer(JNIEnv * env, jobject thisObj,
                                                                                                                   jlong numTableAddr,
                                                                                                                   jlong vectorIndex, jlong vectorNum,
                                                                                                                   jobject byteBuffer)
{
    using namespace daal;
    RowMergedNumericTable * nt = (*(RowMergedNumericTablePtr *)numTableAddr).get();
    BlockDescriptor<float> block;

    const size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfRows(vectorIndex, vectorNum, writeOnly, block));

    float * data = block.getBlockPtr();

    const float * src = (float *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum * nCols; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
}

/*
 * Class:     com_intel_daal_data_1management_data_RowMergedNumericTableImpl
 * Method:    releaseDoubleBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_releaseDoubleBlockBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    RowMergedNumericTable * nt = (*(RowMergedNumericTablePtr *)numTableAddr).get();
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
 * Class:     com_intel_daal_data_1management_data_RowMergedNumericTableImpl
 * Method:    releaseIntBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_releaseIntBlockBuffer(JNIEnv * env, jobject thisObj,
                                                                                                                 jlong numTableAddr,
                                                                                                                 jlong vectorIndex, jlong vectorNum,
                                                                                                                 jobject byteBuffer)
{
    using namespace daal;
    RowMergedNumericTable * nt = (*(RowMergedNumericTablePtr *)numTableAddr).get();
    BlockDescriptor<int> block;

    const size_t nCols = nt->getNumberOfColumns();
    DAAL_CHECK_THROW(nt->getBlockOfRows(vectorIndex, vectorNum, writeOnly, block));

    int * data = block.getBlockPtr();

    const int * src = (int *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum * nCols; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
}

/*
 * Class:     com_intel_daal_data_1management_data_RowMergedNumericTableImpl
 * Method:    getDoubleBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_getDoubleBlockBuffer(JNIEnv * env, jobject thisObj,
                                                                                                                   jlong numTableAddr,
                                                                                                                   jlong vectorIndex, jlong vectorNum,
                                                                                                                   jobject byteBuffer)
{
    using namespace daal;
    RowMergedNumericTable * nt = (*(RowMergedNumericTablePtr *)numTableAddr).get();
    BlockDescriptor<double> block;

    size_t nCols = nt->getNumberOfColumns();
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
 * Class:     com_intel_daal_data_1management_data_RowMergedNumericTableImpl
 * Method:    getFloatBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_getFloatBlockBuffer(JNIEnv * env, jobject thisObj,
                                                                                                                  jlong numTableAddr,
                                                                                                                  jlong vectorIndex, jlong vectorNum,
                                                                                                                  jobject byteBuffer)
{
    using namespace daal;
    RowMergedNumericTable * nt = (*(RowMergedNumericTablePtr *)numTableAddr).get();
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
 * Class:     com_intel_daal_data_1management_data_RowMergedNumericTableImpl
 * Method:    getIntBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_getIntBlockBuffer(JNIEnv * env, jobject thisObj,
                                                                                                                jlong numTableAddr, jlong vectorIndex,
                                                                                                                jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    RowMergedNumericTable * nt = (*(RowMergedNumericTablePtr *)numTableAddr).get();
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
 * Class:     com_intel_daal_data_1management_data_RowMergedNumericTableImpl
 * Method:    getDoubleColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_getDoubleColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    RowMergedNumericTable * nt = (*(RowMergedNumericTablePtr *)numTableAddr).get();
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
 * Class:     com_intel_daal_data_1management_data_RowMergedNumericTableImpl
 * Method:    getFloatColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_getFloatColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    RowMergedNumericTable * nt = (*(RowMergedNumericTablePtr *)numTableAddr).get();
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
 * Class:     com_intel_daal_data_1management_data_RowMergedNumericTableImpl
 * Method:    getIntColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_getIntColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    RowMergedNumericTable * nt = (*(RowMergedNumericTablePtr *)numTableAddr).get();
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
 * Class:     com_intel_daal_data_1management_data_RowMergedNumericTableImpl
 * Method:    releaseFloatColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_releaseFloatColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    RowMergedNumericTable * nt = (*(RowMergedNumericTablePtr *)numTableAddr).get();
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
 * Class:     com_intel_daal_data_1management_data_RowMergedNumericTableImpl
 * Method:    releaseDoubleColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_releaseDoubleColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    RowMergedNumericTable * nt = (*(RowMergedNumericTablePtr *)numTableAddr).get();
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
 * Class:     com_intel_daal_data_1management_data_RowMergedNumericTableImpl
 * Method:    releaseIntColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_releaseIntColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    using namespace daal;
    RowMergedNumericTable * nt = (*(RowMergedNumericTablePtr *)numTableAddr).get();
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
