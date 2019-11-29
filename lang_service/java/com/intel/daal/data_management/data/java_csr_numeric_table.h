/* file: java_csr_numeric_table.h */
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

/*
//++
//  Implementation of the class that connects Java and C++ CSR Numeric Table
//--
*/

#ifndef __JAVA_CSR_NUMERIC_TABLE_H__
#define __JAVA_CSR_NUMERIC_TABLE_H__

#include <jni.h>
#include <tbb/tbb.h>

#include "java_numeric_table.h"
#include "csr_numeric_table.h"
#include "common_defines.i"

using namespace daal::data_management;

namespace daal
{
/**
 *  \brief Class that implements C++ to Java "connector" for CSR Numeric Table.
 *  Getters and Setters of this class are callbacks
 *  to the corresponding methods of user-defined Java class.
 */
class JavaCSRNumericTable : public JavaNumericTable<SERIALIZATION_JAVANIO_CSR_NT_ID>, virtual public CSRNumericTableIface
{
public:
    JavaCSRNumericTable() : JavaNumericTable<SERIALIZATION_JAVANIO_CSR_NT_ID>(DictionaryIface::equal) {}
    /**
     *  Constructor
     *
     *  \param featnum[in]       Number of features
     *  \param obsnum[in]        Number of observations
     *  \param _jvm[in]          Java VM interface function table
     *  \param _JavaNumTable[in] Java object associated with this C++ object
     */
    JavaCSRNumericTable(size_t featnum, size_t obsnum, JavaVM * _jvm, jobject _JavaNumTable)
        : JavaNumericTable(featnum, obsnum, _jvm, _JavaNumTable, StorageLayout::csrArray, DictionaryIface::equal)
    {}

    virtual ~JavaCSRNumericTable() {}

    services::Status getSparseBlock(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, CSRBlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getSparseTBlock<double>(vector_idx, vector_num, rwflag, block, "getDoubleSparseBlock");
    }
    services::Status getSparseBlock(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, CSRBlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getSparseTBlock<float>(vector_idx, vector_num, rwflag, block, "getFloatSparseBlock");
    }
    services::Status getSparseBlock(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, CSRBlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getSparseTBlock<int>(vector_idx, vector_num, rwflag, block, "getIntSparseBlock");
    }

    services::Status releaseSparseBlock(CSRBlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseSparseTBlock<double>(block); }
    services::Status releaseSparseBlock(CSRBlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseSparseTBlock<float>(block); }
    services::Status releaseSparseBlock(CSRBlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseSparseTBlock<int>(block); }

    services::Status releaseBlockOfRows(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTBlock<double>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTBlock<float>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTBlock<int>(block); }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTFeature<double>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTFeature<float>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTFeature<int>(block); }

    size_t getDataSize() DAAL_C11_OVERRIDE
    {
        daal::internal::_java_tls local_tls = tls.local();

        /* Get JNI interface pointer for current thread */
        this->_status |= daal::internal::attachCurrentThread(jvm, local_tls);
        if (!this->_status) return 0;

        /* Get class associated with Java object */
        local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaNumTable);
        if (local_tls.jcls == NULL)
        {
            return 0;
        }

        /* Get ID of the 'getBlockOfRows' method of the Java class */
        jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, "getSparseBlockSize", "(JJ)J");
        if (jmeth == NULL)
        {
            return 0;
        }

        size_t bufferSize = (size_t)((local_tls.jenv)->CallObjectMethod(jJavaNumTable, jmeth, (jlong)0, (jlong)getNumberOfRows()));

        this->_status |= daal::internal::detachCurrentThread(jvm, local_tls);
        if (!this->_status) return bufferSize;

        tls.local() = local_tls;

        return bufferSize;
    }

protected:
    template <typename T>
    services::Status getSparseTBlock(size_t idx, size_t nrows, int rwFlag, CSRBlockDescriptor<T> & block, const char * javaMethodName)
    {
        services::Status status;
        daal::internal::_java_tls local_tls = tls.local();

        /* Get JNI interface pointer for current thread */
        DAAL_CHECK_STATUS(status, daal::internal::attachCurrentThread(jvm, local_tls))

        /* Get class associated with Java object */
        local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaNumTable);
        if (local_tls.jcls == NULL)
        {
            return services::Status();
        }

        /* Get ID of the 'getBlockOfRows' method of the Java class */
        jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, "getSparseBlockSize", "(JJ)J");
        if (jmeth == NULL)
        {
            return services::Status();
        }
        size_t nValues = (size_t)((local_tls.jenv)->CallObjectMethod(jJavaNumTable, jmeth, (jlong)idx, (jlong)nrows));

        size_t ncols = _ddict->getNumberOfFeatures();

        block.setDetails(ncols, idx, rwFlag);

        size_t valuesSize     = nValues * sizeof(T);
        size_t colindicesSize = nValues * sizeof(size_t);
        size_t rowOffsetsSize = (nrows + 1) * sizeof(size_t);

        size_t colIndicesSizeJava = nValues * sizeof(__int64);
        size_t rowOffsetsSizeJava = (nrows + 1) * sizeof(__int64);

        T * valuesBuf               = (T *)daal::services::daal_malloc(valuesSize);
        size_t * colIndicesBuf      = (size_t *)daal::services::daal_malloc(colindicesSize);
        size_t * rowOffsetsBuf      = (size_t *)daal::services::daal_malloc(rowOffsetsSize);
        __int64 * colIndicesBufJava = (__int64 *)colIndicesBuf;
        __int64 * rowOffsetsBufJava = (__int64 *)rowOffsetsBuf;

        if (sizeof(size_t) != sizeof(__int64))
        {
            rowOffsetsBufJava = (__int64 *)daal::services::daal_malloc(rowOffsetsSizeJava);
            colIndicesBufJava = (__int64 *)daal::services::daal_malloc(colIndicesSizeJava);
        }

        jobject jdata       = (local_tls.jenv)->NewDirectByteBuffer(valuesBuf, valuesSize);
        jobject jColIndices = (local_tls.jenv)->NewDirectByteBuffer(colIndicesBufJava, colIndicesSizeJava);
        jobject jRowOffsets = (local_tls.jenv)->NewDirectByteBuffer(rowOffsetsBufJava, rowOffsetsSizeJava);

        jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, javaMethodName, "(JJLjava/nio/ByteBuffer;Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;)J");
        if (jmeth == NULL)
        {
            return services::Status();
        }
        size_t nRows = (size_t)((local_tls.jenv)->CallObjectMethod(jJavaNumTable, jmeth, (jlong)idx, (jlong)nrows, jdata, jColIndices, jRowOffsets));

        if (nRows == 0)
        {
            return services::Status();
        }

        if (sizeof(size_t) != sizeof(__int64))
        {
            for (size_t i = 0; i < nValues; i++)
            {
                colIndicesBuf[i] = (size_t)colIndicesBufJava[i];
            }
            for (size_t i = 0; i < nrows + 1; i++)
            {
                rowOffsetsBuf[i] = (size_t)rowOffsetsBufJava[i];
            }
            daal::services::daal_free(rowOffsetsBufJava);
            daal::services::daal_free(colIndicesBufJava);
        }

        block.setValuesPtr(valuesBuf, nValues);
        block.setColumnIndicesPtr(colIndicesBuf, nValues);
        block.setRowIndicesPtr(rowOffsetsBuf, nrows);

        DAAL_CHECK_STATUS(status, daal::internal::detachCurrentThread(jvm, local_tls))

        tls.local() = local_tls;

        return status;
    }

    template <typename T>
    services::Status releaseSparseTBlock(CSRBlockDescriptor<T> & block)
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            return services::Status(services::ErrorMethodNotSupported);
        }
        void * valuesBuf     = block.getBlockValuesPtr();
        void * colIndicesBuf = block.getBlockColumnIndicesPtr();
        void * rowOffsetsBuf = block.getBlockRowIndicesPtr();

        if (valuesBuf) daal::services::daal_free(valuesBuf);
        if (colIndicesBuf) daal::services::daal_free(colIndicesBuf);
        if (rowOffsetsBuf) daal::services::daal_free(rowOffsetsBuf);

        block.setValuesPtr(0, 0);
        block.setColumnIndicesPtr(0, 0);
        block.setRowIndicesPtr(0, 0);

        block.reset();
        return services::Status();
    }

    template <typename T>
    services::Status releaseTBlock(BlockDescriptor<T> & block)
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            return services::Status(services::ErrorMethodNotSupported);
        }
        block.reset();
        return services::Status();
    }

    template <typename T>
    services::Status releaseTFeature(BlockDescriptor<T> & block)
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            return services::Status(services::ErrorMethodNotSupported);
        }
        block.reset();
        return services::Status();
    }

}; // class JavaCSRNumericTable

} // namespace daal

#endif
