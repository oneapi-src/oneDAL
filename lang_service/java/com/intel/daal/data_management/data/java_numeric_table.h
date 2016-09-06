/* file: java_numeric_table.h */
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

/*
//++
//  Implementation of the class that connects Java and C++ Numeric Tables
//--
*/

#ifndef __JAVA_NUMERIC_TABLE_H__
#define __JAVA_NUMERIC_TABLE_H__

#include <jni.h>
#include <tbb/tbb.h>


#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{

/**
 *  \brief Class that implements C++ to Java "connector".
 *  Getters and Setters of this class are callbacks
 *  to the corresponding methods of user-defined Java class.
 */
class JavaNumericTable : public NumericTable
{
public:
    explicit JavaNumericTable(bool featuresEqual = false): NumericTable(0, 0, featuresEqual), jvm(NULL), jJavaNumTable(NULL) {
    }

    /**
     *  Constructor
     *
     *  \param featnum[in]       Number of features
     *  \param obsnum[in]        Number of observations
     *  \param _jvm[in]          Java VM interface function table
     *  \param _JavaNumTable[in] Java object associated with this C++ object
     */
    JavaNumericTable(size_t featnum, size_t obsnum, JavaVM *_jvm, jobject _JavaNumTable,
                     StorageLayout layout = layout_unknown, bool featuresEqual = false):
        NumericTable(featnum, obsnum, featuresEqual), jvm(_jvm)
    {
        _layout = layout;
        _memStatus = userAllocated;

        _tls local_tls = tls.local();

        /* mark current thread as 'main' in order not to detach it further */
        local_tls.is_main_thread = true;

        jint status = jvm->AttachCurrentThread((void **)(&(local_tls.jenv)), NULL);
        if(status != JNI_OK)
        {
            this->_errors->add(services::ErrorCouldntAttachCurrentThreadToJavaVM);
        }
        else
        {
            local_tls.is_attached = true;
        }

        jJavaNumTable = (local_tls.jenv)->NewGlobalRef(_JavaNumTable);
        if(jJavaNumTable == NULL)
        {
            this->_errors->add(services::ErrorCouldntCreateGlobalReferenceToJavaObject);
        }

        tls.local() = local_tls;
    }

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return SERIALIZATION_JAVANIO_NT_ID;
    }

    /**
     *  Destructor
     */
    virtual ~JavaNumericTable()
    {
        if(jvm != NULL)
        {
            _tls local_tls = tls.local();
            jint status = jvm->AttachCurrentThread((void **)(&(local_tls.jenv)), NULL);
            if(status != JNI_OK)
            {
                this->_errors->add(services::ErrorCouldntAttachCurrentThreadToJavaVM);
            }
            else
            {
                local_tls.is_attached = true;
            }

            if (jJavaNumTable != NULL)
            {
                (local_tls.jenv)->DeleteGlobalRef(jJavaNumTable);
            }
        }
    }

    void serializeImpl (InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<InputDataArchive, false>(arch);}

    void deserializeImpl(OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<OutputDataArchive, true>(arch);}

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        NumericTable::serialImpl<Archive, onDeserialize>(arch);

        if (!onDeserialize)
        {
            _tls local_tls = tls.local();

            local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaNumTable);
            jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, "pack", "()V");
            (local_tls.jenv)->CallObjectMethod(jJavaNumTable, jmeth);

            jclass clsByteArrayOutputStream = (local_tls.jenv)->FindClass("java/io/ByteArrayOutputStream");
            jmethodID byteArrayOutputStreamConstructor = (local_tls.jenv)->GetMethodID(clsByteArrayOutputStream, "<init>", "()V");
            jobject jOutputByteStream = (local_tls.jenv)->NewObject(clsByteArrayOutputStream, byteArrayOutputStreamConstructor);

            jclass clsObjectOutputStream = (local_tls.jenv)->FindClass("java/io/ObjectOutputStream");
            jmethodID objectOutputStreamConstructor = (local_tls.jenv)->GetMethodID(clsObjectOutputStream, "<init>", "(Ljava/io/OutputStream;)V");
            jobject jOutputStream = (local_tls.jenv)->NewObject(clsObjectOutputStream, objectOutputStreamConstructor, jOutputByteStream);

            jmeth = (local_tls.jenv)->GetMethodID(clsObjectOutputStream, "writeObject", "(Ljava/lang/Object;)V");
            (local_tls.jenv)->CallObjectMethod(jOutputStream, jmeth, jJavaNumTable);

            jmeth = (local_tls.jenv)->GetMethodID(clsByteArrayOutputStream, "toByteArray", "()[B");
            jbyteArray jbuf = static_cast<jbyteArray>((local_tls.jenv)->CallObjectMethod(jOutputByteStream, jmeth));

            jbyte *ptr = (local_tls.jenv)->GetByteArrayElements(jbuf, 0);
            size_t size = (local_tls.jenv)->GetArrayLength(jbuf);

            arch->set(_layout);

            arch->set(size);

            arch->set(ptr, size);

            (local_tls.jenv)->ReleaseByteArrayElements(jbuf, ptr, 0);

            tls.local() = local_tls;
        }
        else
        {
            _tls local_tls = tls.local();

            if (jvm == NULL)
            {
                jvm = getJavaVM();
                local_tls.is_main_thread = true;
                jint status = jvm->AttachCurrentThread((void **)(&(local_tls.jenv)), NULL);
                if(status != JNI_OK)
                {
                    this->_errors->add(services::ErrorCouldntAttachCurrentThreadToJavaVM);
                }
                else
                {
                    local_tls.is_attached = true;
                }
            }

            jbyte *ptr;
            size_t size;

            arch->set(_layout);

            arch->set(size);

            ptr = (jbyte *)daal::services::daal_malloc( size * sizeof(jbyte) );
            arch->set(ptr, size);
            jbyteArray jbuf = (local_tls.jenv)->NewByteArray(size);
            (local_tls.jenv)->SetByteArrayRegion(jbuf, 0, size, ptr);

            jclass clsByteArrayInputStream = (local_tls.jenv)->FindClass("java/io/ByteArrayInputStream");
            jmethodID byteArrayInputStreamConstructor = (local_tls.jenv)->GetMethodID(clsByteArrayInputStream, "<init>", "([B)V");
            jobject jInputByteStream = (local_tls.jenv)->NewObject(clsByteArrayInputStream, byteArrayInputStreamConstructor, jbuf);

            jclass clsObjectInputStream = (local_tls.jenv)->FindClass("java/io/ObjectInputStream");
            jmethodID objectInputStreamConstructor = (local_tls.jenv)->GetMethodID(clsObjectInputStream, "<init>", "(Ljava/io/InputStream;)V");
            jobject jInputStream = (local_tls.jenv)->NewObject(clsObjectInputStream, objectInputStreamConstructor, jInputByteStream);

            jmethodID jmeth = (local_tls.jenv)->GetMethodID(clsObjectInputStream, "readObject", "()Ljava/lang/Object;");
            jobject javaNumTable = (local_tls.jenv)->CallObjectMethod(jInputStream, jmeth);

            jclass jNumericTable = (local_tls.jenv)->GetObjectClass(javaNumTable);

            jmethodID numericTableUnpack = (local_tls.jenv)->GetMethodID(jNumericTable, "unpack", "(Lcom/intel/daal/services/DaalContext;)V");
            jobject context = getDaalContext();
            (local_tls.jenv)->CallObjectMethod(javaNumTable, numericTableUnpack, context);
            setDaalContext(context);

            jJavaNumTable = (local_tls.jenv)->NewGlobalRef(javaNumTable);
            if(jJavaNumTable == NULL)
            {
                this->_errors->add(services::ErrorCouldntCreateGlobalReferenceToJavaObject);
            }
            daal::services::daal_free(ptr);
            tls.local() = local_tls;
        }
    }


    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block,
        "getDoubleBlock", "(JJLjava/nio/ByteBuffer;)Ljava/nio/DoubleBuffer;");
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block,
        "getFloatBlock", "(JJLjava/nio/ByteBuffer;)Ljava/nio/FloatBuffer;");
    }
    void getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block,
        "getIntBlock", "(JJLjava/nio/ByteBuffer;)Ljava/nio/IntBuffer;");
    }


    void releaseBlockOfRows(BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<double>(block, "releaseDoubleBlock");
    }
    void releaseBlockOfRows(BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<float>(block, "releaseFloatBlock");
    }
    void releaseBlockOfRows(BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<int>(block, "releaseIntBlock");
    }


    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                ReadWriteMode rwflag, BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block,
        "getDoubleFeature", "(JJJLjava/nio/ByteBuffer;)Ljava/nio/DoubleBuffer;");
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                ReadWriteMode rwflag, BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block,
        "getFloatFeature", "(JJJLjava/nio/ByteBuffer;)Ljava/nio/FloatBuffer;");
    }
    void getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                ReadWriteMode rwflag, BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block,
        "getIntFeature", "(JJJLjava/nio/ByteBuffer;)Ljava/nio/IntBuffer;");
    }


    void releaseBlockOfColumnValues(BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<double>(block, "releaseDoubleFeature");
    }
    void releaseBlockOfColumnValues(BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<float>(block, "releaseFloatFeature");
    }
    void releaseBlockOfColumnValues(BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<int>(block, "releaseIntFeature");
    }


    template<typename T>
    void getTBlock(size_t idx, size_t nrows, ReadWriteMode rwFlag, BlockDescriptor<T> &block,
                   const char *javaMethodName, const char *javaMethodSignature)
    {
        jint status = JNI_OK;
        _tls local_tls = tls.local();

        block.setDetails( 0, idx, rwFlag );

        /* Get JNI interface pointer for current thread */
        status = jvm->AttachCurrentThread((void **)(&(local_tls.jenv)), NULL);
        if(status != JNI_OK) { this->_errors->add(services::ErrorCouldntAttachCurrentThreadToJavaVM); return; }

        local_tls.is_attached = true;

        size_t ncols = _ddict->getNumberOfFeatures();
        size_t bufferSize = nrows * ncols * sizeof(T);

        if( !block.resizeBuffer( ncols, nrows ) ) { return; }

        void *buf = block.getBlockPtr();

        /* Get class associated with Java object */
        local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaNumTable);
        if(local_tls.jcls == NULL)
        {
            this->_errors->add(services::ErrorCouldntFindClassForJavaObject);
            return;
        }

        /* Get ID of the 'getBlockOfRows' method of the Java class */
        jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, javaMethodName,
                                                        javaMethodSignature);
        if(jmeth == NULL)
        {
            services::SharedPtr<services::Error> e(new services::Error(services::ErrorCouldntFindJavaMethod));
            e->addStringDetail(services::Method, services::String(javaMethodName));
            this->_errors->add(e);
            return;
        }

        /* Call 'getBlockOfRows' Java method */
        local_tls.jbuf = (local_tls.jenv)->NewDirectByteBuffer( buf, bufferSize);
        local_tls.jbuf = (local_tls.jenv)->CallObjectMethod(jJavaNumTable, jmeth, (jlong)idx, (jlong)nrows, local_tls.jbuf);

        buf = (local_tls.jenv)->GetDirectBufferAddress(local_tls.jbuf);

        tls.local() = local_tls;

        block.setPtr( (T *)buf, ncols, nrows );

        if(!local_tls.is_main_thread)
        {
            status = jvm->DetachCurrentThread();
            if(status != JNI_OK)
            {
                this->_errors->add(services::ErrorCouldntAttachCurrentThreadToJavaVM);
                return;
            }
        }
    }

    template<typename T>
    void releaseTBlock(size_t idx, size_t nrows, void *buf, ReadWriteMode rwFlag,
                       const char *javaMethodName)
    {
        jint status = JNI_OK;
        _tls local_tls = tls.local();
        if(rwFlag == writeOnly)
        {
            /* Get JNI interface pointer for current thread */
            status = jvm->AttachCurrentThread((void **)(&(local_tls.jenv)), NULL);
            if(status != JNI_OK)
            {
                this->_errors->add(services::ErrorCouldntAttachCurrentThreadToJavaVM);
            }

            /* Get class associated with Java object */
            local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaNumTable);
            if(local_tls.jcls == NULL)
            {
                this->_errors->add(services::ErrorCouldntFindClassForJavaObject);
            }

            /* Get ID of the 'releaseBlockOfRows' method of the Java class */
            jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, javaMethodName,
                                                            "(JJLjava/nio/ByteBuffer;)V");
            if(jmeth == NULL)
            {
                services::SharedPtr<services::Error> e(new services::Error(services::ErrorCouldntFindJavaMethod));
                e->addStringDetail(services::Method, services::String(javaMethodName));
                this->_errors->add(e);
            }

            size_t ncols = _ddict->getNumberOfFeatures();
            size_t bufferSize = nrows * ncols * sizeof(T);
            local_tls.jbuf = (local_tls.jenv)->NewDirectByteBuffer(buf, bufferSize);

            /* Call 'releaseBlockOfRows' Java method */
            (local_tls.jenv)->CallObjectMethod(
                jJavaNumTable, jmeth, (jlong)idx, (jlong)nrows, local_tls.jbuf, rwFlag);

            if(!local_tls.is_main_thread)
            {
                status = jvm->DetachCurrentThread();
                if(status != JNI_OK)
                {
                    this->_errors->add(services::ErrorCouldntAttachCurrentThreadToJavaVM);
                }
            }
        }

        tls.local() = local_tls;
    }

    template<typename T>
    void releaseTBlock(BlockDescriptor<T> &block, const char *javaMethodName)
    {
        jint status = JNI_OK;
        _tls local_tls = tls.local();
        if(block.getRWFlag() == writeOnly)
        {
            /* Get JNI interface pointer for current thread */
            status = jvm->AttachCurrentThread((void **)(&(local_tls.jenv)), NULL);
            if(status != JNI_OK)
            {
                this->_errors->add(services::ErrorCouldntAttachCurrentThreadToJavaVM);
                return;
            }

            /* Get class associated with Java object */
            local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaNumTable);
            if(local_tls.jcls == NULL)
            {
                this->_errors->add(services::ErrorCouldntFindClassForJavaObject);
                return;
            }

            /* Get ID of the 'releaseBlockOfRows' method of the Java class */
            jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, javaMethodName, "(JJLjava/nio/ByteBuffer;)V");
            if(jmeth == NULL)
            {
                services::SharedPtr<services::Error> e(new services::Error(services::ErrorCouldntFindJavaMethod));
                e->addStringDetail(services::Method, services::String(javaMethodName));
                this->_errors->add(e);
                return;
            }

            size_t idx   = block.getRowsOffset();
            size_t nrows = block.getNumberOfRows();
            size_t ncols = _ddict->getNumberOfFeatures();
            size_t bufferSize = nrows * ncols * sizeof(T);
            local_tls.jbuf = (local_tls.jenv)->NewDirectByteBuffer( block.getBlockPtr(), bufferSize);

            /* Call 'releaseBlockOfRows' Java method */
            (local_tls.jenv)->CallObjectMethod(
                jJavaNumTable, jmeth, (jlong)idx, (jlong)nrows, local_tls.jbuf, block.getRWFlag());

            if(!local_tls.is_main_thread)
            {
                status = jvm->DetachCurrentThread();
                if(status != JNI_OK)
                {
                    this->_errors->add(services::ErrorCouldntAttachCurrentThreadToJavaVM);
                    return;
                }
            }
        }

        tls.local() = local_tls;
        block.setDetails( 0, 0, 0 );
    }


    template<typename T>
    void getTFeature(size_t feature_idx, size_t idx, size_t nrows, ReadWriteMode rwFlag, BlockDescriptor<T> &block,
                     const char *javaMethodName, const char *javaMethodSignature)
    {
        jint status = JNI_OK;
        _tls local_tls = tls.local();

        block.setDetails( feature_idx, idx, rwFlag );

        /* Get JNI interface pointer for current thread */
        status = jvm->AttachCurrentThread((void **)(&(local_tls.jenv)), NULL);
        if(status != JNI_OK)
        {
            this->_errors->add(services::ErrorCouldntAttachCurrentThreadToJavaVM);
            return;
        }

        local_tls.is_attached = true;

        size_t bufferSize = nrows * sizeof(T);

        if( !block.resizeBuffer( 1, nrows ) ) { return; }

        void *buf = block.getBlockPtr();

        /* Get class associated with Java object */
        local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaNumTable);
        if(local_tls.jcls == NULL)
        {
            this->_errors->add(services::ErrorCouldntFindClassForJavaObject);
            return;
        }

        /* Get ID of the 'getBlockOfRows' method of the Java class */
        jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, javaMethodName,
                                                        javaMethodSignature);
        if(jmeth == NULL)
        {
            services::SharedPtr<services::Error> e(new services::Error(services::ErrorCouldntFindJavaMethod));
            e->addStringDetail(services::Method, services::String(javaMethodName));
            this->_errors->add(e);
            return;
        }

        /* Call 'getBlockOfRows' Java method */
        local_tls.jbuf = (local_tls.jenv)->NewDirectByteBuffer(buf, bufferSize);

        local_tls.jbuf = (local_tls.jenv)->CallObjectMethod(
                             jJavaNumTable, jmeth, (jlong)feature_idx, (jlong)idx, (jlong)nrows, local_tls.jbuf);

        buf = (local_tls.jenv)->GetDirectBufferAddress(local_tls.jbuf);

        tls.local() = local_tls;

        block.setPtr( (T *)buf, 1, nrows );
    }

    template<typename T>
    void releaseTFeature(size_t feature_idx, size_t idx, size_t nrows, void *buf, ReadWriteMode rwFlag,
                         const char *javaMethodName)
    {
        jint status = JNI_OK;
        if(rwFlag == writeOnly)
        {
            _tls local_tls = tls.local();
            /* Get JNI interface pointer for current thread */
            status = jvm->AttachCurrentThread((void **)(&(local_tls.jenv)), NULL);
            if(status != JNI_OK)
            {
                this->_errors->add(services::ErrorCouldntAttachCurrentThreadToJavaVM);
            }

            /* Get class associated with Java object */
            local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaNumTable);
            if(local_tls.jcls == NULL)
            {
                this->_errors->add(services::ErrorCouldntFindClassForJavaObject);
            }

            /* Get ID of the 'releaseBlockOfRows' method of the Java class */
            jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, javaMethodName,
                                                            "(JJJLjava/nio/ByteBuffer;)V");
            if(jmeth == NULL)
            {
                services::SharedPtr<services::Error> e(new services::Error(services::ErrorCouldntFindJavaMethod));
                e->addStringDetail(services::Method, services::String(javaMethodName));
                this->_errors->add(e);
            }

            size_t bufferSize = nrows * sizeof(T);
            local_tls.jbuf = (local_tls.jenv)->NewDirectByteBuffer(buf, bufferSize);

            /* Call 'releaseBlockOfRows' Java method */
            (local_tls.jenv)->CallObjectMethod(
                jJavaNumTable, jmeth, (jlong)feature_idx, (jlong)idx, (jlong)nrows, local_tls.jbuf, rwFlag);

            tls.local() = local_tls;
        }
    }

    template<typename T>
    void releaseTFeature(BlockDescriptor<T> &block, const char *javaMethodName)
    {
        jint status = JNI_OK;
        if(block.getRWFlag() == writeOnly)
        {
            _tls local_tls = tls.local();
            /* Get JNI interface pointer for current thread */
            status = jvm->AttachCurrentThread((void **)(&(local_tls.jenv)), NULL);
            if(status != JNI_OK)
            {
                this->_errors->add(services::ErrorCouldntAttachCurrentThreadToJavaVM);
                return;
            }

            /* Get class associated with Java object */
            local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaNumTable);
            if(local_tls.jcls == NULL)
            {
                this->_errors->add(services::ErrorCouldntFindClassForJavaObject);
                return;
            }

            /* Get ID of the 'releaseBlockOfRows' method of the Java class */
            jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, javaMethodName,
                                                            "(JJJLjava/nio/ByteBuffer;)V");
            if(jmeth == NULL)
            {
                services::SharedPtr<services::Error> e(new services::Error(services::ErrorCouldntFindJavaMethod));
                e->addStringDetail(services::Method, services::String(javaMethodName));
                this->_errors->add(e);
                return;
            }

            size_t idx   = block.getRowsOffset();
            size_t nrows = block.getNumberOfRows();
            size_t feature_idx = block.getColumnsOffset();

            size_t bufferSize = nrows * sizeof(T);
            local_tls.jbuf = (local_tls.jenv)->NewDirectByteBuffer( block.getBlockPtr(), bufferSize);

            /* Call 'releaseBlockOfRows' Java method */
            (local_tls.jenv)->CallObjectMethod( jJavaNumTable, jmeth, (jlong)feature_idx, (jlong)idx, (jlong)nrows,
                                                local_tls.jbuf, block.getRWFlag());

            tls.local() = local_tls;
        }
        block.setDetails( 0, 0, 0 );
    }

    virtual void allocateDataMemory(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE {}

    virtual void freeDataMemory() DAAL_C11_OVERRIDE {}

    static void setJavaVM(JavaVM *jvm)
    {
        if (globalJavaVM == NULL)
        {
            globalJavaVM = jvm;
            Factory::instance().registerObject(new Creator<JavaNumericTable>());
        }
    }

    static JavaVM *getJavaVM()
    {
        return globalJavaVM;
    }

    static void setDaalContext(jobject context)
    {
        globalDaalContext.local() = context;
    }

    static jobject getDaalContext()
    {
        return globalDaalContext.local();
    }

    jobject getJavaObject() { return jJavaNumTable; }
protected:
    struct _tls
    {
        JNIEnv *jenv;    // JNI interface poiner
        jobject jbuf;
        jclass jcls;     // Java class associated with this C++ object
        bool is_main_thread;
        bool is_attached;
        /* Default constructor */
        _tls()
        {
            jenv = NULL;
            jbuf = NULL;
            jcls = NULL;
            is_main_thread = false;
            is_attached = false;
        }
    };
    tbb::enumerable_thread_specific<_tls> tls;  /**< Thread local storage */
    jobject jJavaNumTable;                      /**< Java object associated with this C++ object */
    JavaVM *jvm;                                /**< Java VM interface function table */

private:
    static JavaVM *globalJavaVM;
    static tbb::enumerable_thread_specific<jobject> globalDaalContext;
};

} // namespace daal

#endif
