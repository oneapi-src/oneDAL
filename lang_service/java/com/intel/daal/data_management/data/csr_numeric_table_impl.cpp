/* file: csr_numeric_table_impl.cpp */
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

/*
//++
//  Implementation of the JNI layer for Java CSR Numeric Table
//--
*/

#include <jni.h>

#include "JCSRNumericTableImpl.h"
#include "java_csr_numeric_table.h"
#include "daal.h"
#include "common_helpers_functions.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_data_CSRNumericTableImpl
 * Method:    initCSRNumericTable
 * Signature:(JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_initCSRNumericTable
(JNIEnv *env, jobject thisObj, jlong nFeatures, jlong nVectors)
{
    JavaVM *jvm;
    // Get pointer to the Java VM interface function table
    jint status = env->GetJavaVM(&jvm);
    if(status != 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), "Error: Couldn't get Java VM");
        return 0;
    }

    // Create C++ object of the class JavaNumericTable
    JavaCSRNumericTable *tbl = new JavaCSRNumericTable((size_t)nFeatures, (size_t)nVectors, jvm, thisObj);

    if(tbl->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tbl->getErrors()->getDescription());
    }

    return (jlong)new SerializationIfacePtr(tbl);
}

/*
 * Class:     com_intel_daal_data_1management_data_CSRNumericTableImpl
 * Method:    cGetNumberOfRows
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_cGetNumberOfRows
(JNIEnv *env, jobject thisobj, jlong numTableAddr)
{
    NumericTable *nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());

    return(jint)(nt->getNumberOfRows());
}

/*
 * Class:     com_intel_daal_data_1management_data_CSRNumericTableImpl
 * Method:    cGetDataSize
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_cGetDataSize
(JNIEnv *env, jobject thisobj, jlong numTableAddr)
{
    CSRNumericTable *nt = static_cast<CSRNumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());

    return(jint)(nt->getDataSize());
}

/*
 * Class:     com_intel_daal_data_1management_data_CSRNumericTableImpl
 * Method:    getIndexType
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_getIndexType
(JNIEnv *env, jobject thisobj, jlong numTableAddr)
{
    NumericTable *nt = static_cast<NumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());
    NumericTableDictionary *dict = nt->getDictionary();
    return(jint)((*dict)[0].indexType);
}

/*
 * Class:     com_intel_daal_data_management_data_CSRNumericTableImpl
 * Method:    getColIndicesBuffer
 * Signature: (JLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_getColIndicesBuffer
(JNIEnv *env, jobject thisobj, jlong numTableAddr, jobject byteBuffer)
{
    CSRNumericTable *nt = static_cast<CSRNumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());

    size_t dataSize = nt->getDataSize();
    double *ptr;
    size_t *colIndices;
    size_t *rowOffsets;
    nt->getArrays<double>(&ptr, &colIndices, &rowOffsets); //template parameter doesn't matter

    __int64 *dest = (__int64 *)(env->GetDirectBufferAddress(byteBuffer));

    for(size_t i = 0; i < dataSize; i++)
    {
        dest[i] = colIndices[i];
    }
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_management_data_CSRNumericTableImpl
 * Method:    getRowOffsetsBuffer
 * Signature: (JLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_getRowOffsetsBuffer
(JNIEnv *env, jobject thisobj, jlong numTableAddr, jobject byteBuffer)
{
    CSRNumericTable *nt = static_cast<CSRNumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());

    size_t nRows = nt->getNumberOfRows();
    double *ptr;
    size_t *colIndices;
    size_t *rowOffsets;
    nt->getArrays<double>(&ptr, &colIndices, &rowOffsets);//template parameter doesn't matter

    __int64 *dest = (__int64 *)(env->GetDirectBufferAddress(byteBuffer));

    for(size_t i = 0; i < nRows; i++)
    {
        dest[i] = rowOffsets[i];
    }
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_management_data_CSRNumericTableImpl
 * Method:    getDoubleBuffer
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_getDoubleBuffer
(JNIEnv *env, jobject thisobj, jlong numTableAddr)
{
    CSRNumericTable *nt = static_cast<CSRNumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());

    size_t nRows = nt->getNumberOfRows();

    double *ptr;
    size_t *column_indices;
    size_t *row_offsets;

    nt->getArrays<double>(&ptr, &column_indices, &row_offsets);

    size_t dataSize = nt->getDataSize();
    jobject byteBuffer = env->NewDirectByteBuffer(ptr, (jlong)(dataSize * sizeof(double)));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_management_data_CSRNumericTableImpl
 * Method:    getFloatBuffer
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_getFloatBuffer
(JNIEnv *env, jobject thisobj, jlong numTableAddr)
{
    CSRNumericTable *nt = static_cast<CSRNumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());

    size_t nRows = nt->getNumberOfRows();

    float *ptr;
    size_t *column_indices;
    size_t *row_offsets;

    nt->getArrays<float>(&ptr, &column_indices, &row_offsets);

    size_t dataSize = nt->getDataSize();
    jobject byteBuffer = env->NewDirectByteBuffer(ptr, (jlong)(dataSize * sizeof(float)));
    return byteBuffer;
}

/*
 * Class:     com_intel_daal_data_management_data_CSRNumericTableImpl
 * Method:    getLongBuffer
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_CSRNumericTableImpl_getLongBuffer
(JNIEnv *env, jobject thisobj, jlong numTableAddr)
{
    CSRNumericTable *nt = static_cast<CSRNumericTable *>(((SerializationIfacePtr *)numTableAddr)->get());

    size_t nRows = nt->getNumberOfRows();

    int *ptr;
    size_t *column_indices;
    size_t *row_offsets;

    nt->getArrays<int>(&ptr, &column_indices, &row_offsets);

    size_t dataSize = nt->getDataSize();
    jobject byteBuffer = env->NewDirectByteBuffer(ptr, (jlong)(dataSize * sizeof(int)));
    return byteBuffer;
}
