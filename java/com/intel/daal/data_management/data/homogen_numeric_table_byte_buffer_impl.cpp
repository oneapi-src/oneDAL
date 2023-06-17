/* file: homogen_numeric_table_byte_buffer_impl.cpp */
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

#include "com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "com/intel/daal/common_helpers_functions.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    getIndexType
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_getIndexType(JNIEnv * env, jobject thisobj,
                                                                                                                jlong numTableAddr)
{
    NumericTable * nt             = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    NumericTableDictionary * dict = nt->getDictionary();
    return (jint)((*dict)[0].indexType);
}

/*
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    dInit
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_dInit(JNIEnv * env, jobject thisobj,
                                                                                                          jlong nColumns, jint featuresEqual)
{
    HomogenNumericTable<double> * tbl = new HomogenNumericTable<double>((DictionaryIface::FeaturesEqual)featuresEqual, NULL, nColumns, 0);
    SerializationIfacePtr * sPtr      = new SerializationIfacePtr(tbl);

    if (tbl->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tbl->getErrors()->getDescription());
    }
    return (jlong)sPtr;
}

/*
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    sInit
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_sInit(JNIEnv * env, jobject thisobj,
                                                                                                          jlong nColumns, jint featuresEqual)
{
    HomogenNumericTable<float> * tbl = new HomogenNumericTable<float>((DictionaryIface::FeaturesEqual)featuresEqual, NULL, nColumns, 0);
    SerializationIfacePtr * sPtr     = new SerializationIfacePtr(tbl);
    if (tbl->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tbl->getErrors()->getDescription());
    }
    return (jlong)sPtr;
}

/*
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    lInit
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_lInit(JNIEnv * env, jobject thisobj,
                                                                                                          jlong nColumns, jint featuresEqual)
{
    HomogenNumericTable<__int64> * tbl = new HomogenNumericTable<__int64>((DictionaryIface::FeaturesEqual)featuresEqual, NULL, nColumns, 0);
    SerializationIfacePtr * sPtr       = new SerializationIfacePtr(tbl);
    if (tbl->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tbl->getErrors()->getDescription());
    }
    return (jlong)sPtr;
}

/*
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    iInit
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_iInit(JNIEnv * env, jobject thisobj,
                                                                                                          jlong nColumns, jint featuresEqual)
{
    HomogenNumericTable<int> * tbl = new HomogenNumericTable<int>((DictionaryIface::FeaturesEqual)featuresEqual, NULL, nColumns, 0);
    SerializationIfacePtr * sPtr   = new SerializationIfacePtr(tbl);
    if (tbl->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tbl->getErrors()->getDescription());
    }
    return (jlong)sPtr;
}

/*
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    getDoubleBuffer
 * Signature:(J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_getDoubleBuffer(JNIEnv * env, jobject thisObj,
                                                                                                                      jlong numTableAddr)
{
    HomogenNumericTable<double> * nt = static_cast<HomogenNumericTable<double> *>(((SerializationIfacePtr *)numTableAddr)->get());

    size_t nRows  = nt->getNumberOfRows();
    size_t nCols  = nt->getNumberOfColumns();
    double * data = nt->getArray();

    jobject byteBuffer = env->NewDirectByteBuffer(data, ((jlong)nRows * nCols * sizeof(double)));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    getFloatBuffer
 * Signature:(J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_getFloatBuffer(JNIEnv * env, jobject thisObj,
                                                                                                                     jlong numTableAddr)
{
    HomogenNumericTable<float> * nt = static_cast<HomogenNumericTable<float> *>(((SerializationIfacePtr *)numTableAddr)->get());

    size_t nRows = nt->getNumberOfRows();
    size_t nCols = nt->getNumberOfColumns();
    float * data = nt->getArray();

    jobject byteBuffer = env->NewDirectByteBuffer(data, ((jlong)nRows * nCols * sizeof(float)));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    getLongBuffer
 * Signature:(J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_getLongBuffer(JNIEnv * env, jobject thisObj,
                                                                                                                    jlong numTableAddr)
{
    HomogenNumericTable<__int64> * nt = static_cast<HomogenNumericTable<__int64> *>(((SerializationIfacePtr *)numTableAddr)->get());

    size_t nRows   = nt->getNumberOfRows();
    size_t nCols   = nt->getNumberOfColumns();
    __int64 * data = nt->getArray();

    jobject byteBuffer = env->NewDirectByteBuffer(data, ((jlong)nRows * nCols * sizeof(__int64)));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    getIntBuffer
 * Signature:(J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_getIntBuffer(JNIEnv * env, jobject thisObj,
                                                                                                                   jlong numTableAddr)
{
    HomogenNumericTable<int> * nt = static_cast<HomogenNumericTable<int> *>(((SerializationIfacePtr *)numTableAddr)->get());

    size_t nRows = nt->getNumberOfRows();
    size_t nCols = nt->getNumberOfColumns();
    int * data   = nt->getArray();

    jobject byteBuffer = env->NewDirectByteBuffer(data, ((jlong)nRows * nCols * sizeof(int)));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    releaseFloatBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_releaseFloatBlockBuffer(
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
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    releaseDoubleBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_releaseDoubleBlockBuffer(
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
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    releaseIntBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_releaseIntBlockBuffer(
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
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    getDoubleBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_getDoubleBlockBuffer(
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
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    getFloatBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_getFloatBlockBuffer(
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
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    getIntBlockBuffer
 * Signature:(JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_getIntBlockBuffer(
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
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    getDoubleColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_getDoubleColumnBuffer(
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
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    getFloatColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_getFloatColumnBuffer(
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
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    getIntColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_getIntColumnBuffer(
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
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    releaseFloatColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_releaseFloatColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<float> block;

    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, writeOnly, block));

    float * data      = block.getBlockPtr();
    const float * src = (float *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
}

/*
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    releaseDoubleColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_releaseDoubleColumnBuffer(
    JNIEnv * env, jobject thisObj, jlong numTableAddr, jlong featureIndex, jlong vectorIndex, jlong vectorNum, jobject byteBuffer)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<double> block;

    DAAL_CHECK_THROW(nt->getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, writeOnly, block));

    double * data      = block.getBlockPtr();
    const double * src = (double *)(env->GetDirectBufferAddress(byteBuffer));

    for (size_t i = 0; i < vectorNum; i++)
    {
        data[i] = src[i];
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfColumnValues(block));
}

/*
 * Class:     com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl
 * Method:    releaseIntColumnBuffer
 * Signature:(JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_releaseIntColumnBuffer(
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
 * Class:     com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl
 * Method:    assignLong
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_assignLong(JNIEnv * env, jobject,
                                                                                                              jlong numTableAddr, jlong constValue)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<int> block;

    size_t nCols = nt->getNumberOfColumns();
    size_t nRows = nt->getNumberOfRows();

    DAAL_CHECK_THROW(nt->getBlockOfRows(0, nRows, readWrite, block));

    int * data = block.getBlockPtr();

    for (size_t i = 0; i < nRows * nCols; i++)
    {
        data[i] = (int)constValue;
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl
 * Method:    assignInt
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_assignInt(JNIEnv * env, jobject,
                                                                                                             jlong numTableAddr, jint constValue)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<int> block;

    size_t nCols = nt->getNumberOfColumns();
    size_t nRows = nt->getNumberOfRows();

    DAAL_CHECK_THROW(nt->getBlockOfRows(0, nRows, readWrite, block));

    int * data = block.getBlockPtr();

    for (size_t i = 0; i < nRows * nCols; i++)
    {
        data[i] = (int)constValue;
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl
 * Method:    assignDouble
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_assignDouble(JNIEnv * env, jobject,
                                                                                                                jlong numTableAddr,
                                                                                                                jdouble constValue)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<double> block;

    size_t nCols = nt->getNumberOfColumns();
    size_t nRows = nt->getNumberOfRows();

    DAAL_CHECK_THROW(nt->getBlockOfRows(0, nRows, readWrite, block));

    double * data = block.getBlockPtr();

    for (size_t i = 0; i < nRows * nCols; i++)
    {
        data[i] = (double)constValue;
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl
 * Method:    assignFloat
 * Signature: (JF)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_assignFloat(JNIEnv * env, jobject,
                                                                                                               jlong numTableAddr, jfloat constValue)
{
    NumericTable * nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    BlockDescriptor<float> block;

    size_t nCols = nt->getNumberOfColumns();
    size_t nRows = nt->getNumberOfRows();

    DAAL_CHECK_THROW(nt->getBlockOfRows(0, nRows, readWrite, block));

    float * data = block.getBlockPtr();

    for (size_t i = 0; i < nRows * nCols; i++)
    {
        data[i] = (float)constValue;
    }

    DAAL_CHECK_THROW(nt->releaseBlockOfRows(block));
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl
 * Method:    set
 * Signature: (JJD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_cSetDouble(JNIEnv * env, jobject, jlong numTableAddr,
                                                                                                       jlong row, jlong column, jdouble value)
{
    HomogenNumericTable<double> * nt = static_cast<HomogenNumericTable<double> *>(((SerializationIfacePtr *)numTableAddr)->get());
    (*nt)[row][column]               = (double)value;
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl
 * Method:    set
 * Signature: (JJF)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_cSetFloat(JNIEnv * env, jobject, jlong numTableAddr,
                                                                                                       jlong row, jlong column, jfloat value)
{
    HomogenNumericTable<float> * nt = static_cast<HomogenNumericTable<float> *>(((SerializationIfacePtr *)numTableAddr)->get());
    (*nt)[row][column]              = (float)value;
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl
 * Method:    set
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_cSetLong(JNIEnv * env, jobject, jlong numTableAddr,
                                                                                                       jlong row, jlong column, jlong value)
{
    HomogenNumericTable<long> * nt = static_cast<HomogenNumericTable<long> *>(((SerializationIfacePtr *)numTableAddr)->get());
    (*nt)[row][column]             = (long)value;
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl
 * Method:    set
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_cSetInt(JNIEnv * env, jobject, jlong numTableAddr,
                                                                                                       jlong row, jlong column, jint value)
{
    HomogenNumericTable<int> * nt = static_cast<HomogenNumericTable<int> *>(((SerializationIfacePtr *)numTableAddr)->get());
    (*nt)[row][column]            = (int)value;
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl
 * Method:    getDouble
 * Signature: (JJ)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_cGetDouble(JNIEnv * env, jobject,
                                                                                                                jlong numTableAddr, jlong row,
                                                                                                                jlong column)
{
    HomogenNumericTable<double> * nt = static_cast<HomogenNumericTable<double> *>(((SerializationIfacePtr *)numTableAddr)->get());
    return (jdouble)((*nt)[row][column]);
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl
 * Method:    getFloat
 * Signature: (JJ)F
 */
JNIEXPORT jfloat JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_cGetFloat(JNIEnv * env, jobject,
                                                                                                              jlong numTableAddr, jlong row,
                                                                                                              jlong column)
{
    HomogenNumericTable<float> * nt = static_cast<HomogenNumericTable<float> *>(((SerializationIfacePtr *)numTableAddr)->get());
    return (jfloat)((*nt)[row][column]);
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl
 * Method:    getLong
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_cGetLong(JNIEnv * env, jobject, jlong numTableAddr,
                                                                                                            jlong row, jlong column, jlong value)
{
    HomogenNumericTable<long> * nt = static_cast<HomogenNumericTable<long> *>(((SerializationIfacePtr *)numTableAddr)->get());
    return (jlong)((*nt)[row][column]);
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl
 * Method:    getInt
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_cGetInt(JNIEnv * env, jobject, jlong numTableAddr,
                                                                                                          jlong row, jlong column)
{
    HomogenNumericTable<int> * nt = static_cast<HomogenNumericTable<int> *>(((SerializationIfacePtr *)numTableAddr)->get());
    return (jint)((*nt)[row][column]);
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl
 * Method:    cAllocateDataMemoryDouble
 * Signature:(J)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_cAllocateDataMemoryDouble(JNIEnv * env,
                                                                                                                             jobject thisObj,
                                                                                                                             jlong numTableAddr)
{
    NumericTable * tbl = ((NumericTablePtr *)numTableAddr)->get();

    DAAL_CHECK_THROW(((HomogenNumericTable<double> *)tbl)->allocateDataMemory());
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl
 * Method:    cAllocateDataMemoryFloat
 * Signature:(J)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_cAllocateDataMemoryFloat(JNIEnv * env,
                                                                                                                            jobject thisObj,
                                                                                                                            jlong numTableAddr)
{
    NumericTable * tbl = ((NumericTablePtr *)numTableAddr)->get();

    DAAL_CHECK_THROW(((HomogenNumericTable<float> *)tbl)->allocateDataMemory());
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl
 * Method:    cAllocateDataMemoryLong
 * Signature:(J)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_cAllocateDataMemoryLong(JNIEnv * env,
                                                                                                                           jobject thisObj,
                                                                                                                           jlong numTableAddr)
{
    NumericTable * tbl = ((NumericTablePtr *)numTableAddr)->get();

    DAAL_CHECK_THROW(((HomogenNumericTable<long> *)tbl)->allocateDataMemory());
}

/*
 * Class:     com_intel_daal_data_management_data_HomogenNumericTableByteBufferImpl
 * Method:    cAllocateDataMemoryInt
 * Signature:(J)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_HomogenNumericTableByteBufferImpl_cAllocateDataMemoryInt(JNIEnv * env,
                                                                                                                          jobject thisObj,
                                                                                                                          jlong numTableAddr)
{
    NumericTable * tbl = ((NumericTablePtr *)numTableAddr)->get();

    DAAL_CHECK_THROW(((HomogenNumericTable<int> *)tbl)->allocateDataMemory());
}
