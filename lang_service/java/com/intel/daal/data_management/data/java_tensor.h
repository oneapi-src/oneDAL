/* file: java_tensor.h */
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
//  Implementation of the class that connects Java and C++ Tensors
//--
*/

#ifndef __JAVA_TENSOR_H__
#define __JAVA_TENSOR_H__

#include <jni.h>
#include <tbb/tbb.h>


#include "daal.h"


namespace daal
{

using namespace daal::services;
using namespace daal::data_management;

/**
 *  \brief Class that implements C++ to Java "connector".
 *  Getters and Setters of this class are callbacks
 *  to the corresponding methods of user-defined Java class.
 */
class JavaTensor : public Tensor
{
public:
    explicit JavaTensor(): Tensor(&_layout), jvm(NULL), jJavaTensor(NULL), _layout(services::Collection<size_t>()) {}

    /**
     *  Constructor
     *
     *  \param featnum[in]       Number of features
     *  \param obsnum[in]        Number of observations
     *  \param _jvm[in]          Java VM interface function table
     *  \param _JavaTensor[in] Java object associated with this C++ object
     */
    JavaTensor(Collection<size_t> &dims, JavaVM *_jvm, jobject _JavaTensor):
        Tensor(&_layout), jvm(_jvm), _layout(dims)
    {
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

        jJavaTensor = (local_tls.jenv)->NewGlobalRef(_JavaTensor);
        if(jJavaTensor == NULL)
        {
            this->_errors->add(services::ErrorCouldntCreateGlobalReferenceToJavaObject);
        }

        tls.local() = local_tls;
    }

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return SERIALIZATION_JAVANIO_TENSOR_ID;
    }

    /**
     *  Destructor
     */
    virtual ~JavaTensor()
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

            if (jJavaTensor != NULL)
            {
                (local_tls.jenv)->DeleteGlobalRef(jJavaTensor);
            }
        }
    }

    virtual void setDimensions(size_t nDim, const size_t *dimSizes) DAAL_C11_OVERRIDE
    {
        if(!dimSizes)
        {
            this->_errors->add(services::ErrorNullParameterNotSupported);
            return;
        }

        _layout = TensorOffsetLayout(services::Collection<size_t>(nDim, dimSizes));
    }

    virtual void setDimensions(const services::Collection<size_t>& dimensions) DAAL_C11_OVERRIDE
    {
        if(!dimensions.size())
        {
            this->_errors->add(services::ErrorNullParameterNotSupported);
            return;
        }

        _layout = TensorOffsetLayout(dimensions);
    }

    void serializeImpl (InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<InputDataArchive, false>(arch);}

    void deserializeImpl(OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<OutputDataArchive, true>(arch);}

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        //Tensor::serialImpl<Archive, onDeserialize>(arch);
    }

    void getSubtensorEx(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, ReadWriteMode rwflag,
        SubtensorDescriptor<double>& subtensor, const TensorOffsetLayout& layout ) DAAL_C11_OVERRIDE
    {
        return getTSubtensor<double>(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, layout,
        "getDoubleSubtensor", "([JJJLjava/nio/ByteBuffer;)Ljava/nio/DoubleBuffer;");
    }
    void getSubtensorEx(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, ReadWriteMode rwflag,
        SubtensorDescriptor<float>& subtensor, const TensorOffsetLayout& layout ) DAAL_C11_OVERRIDE
    {
        return getTSubtensor<float>(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, layout,
        "getFloatSubtensor", "([JJJLjava/nio/ByteBuffer;)Ljava/nio/FloatBuffer;");
    }
    void getSubtensorEx(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, ReadWriteMode rwflag,
        SubtensorDescriptor<int>& subtensor, const TensorOffsetLayout& layout ) DAAL_C11_OVERRIDE
    {
        return getTSubtensor<int>(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, layout,
        "getIntSubtensor", "([JJJLjava/nio/ByteBuffer;)Ljava/nio/IntBuffer;");
    }


    void releaseSubtensor(SubtensorDescriptor<double> &subtensor) DAAL_C11_OVERRIDE
    {
        releaseTSubtensor<double>(subtensor, "releaseDoubleSubtensor");
    }
    void releaseSubtensor(SubtensorDescriptor<float> &subtensor) DAAL_C11_OVERRIDE
    {
        releaseTSubtensor<float>(subtensor, "releaseFloatSubtensor");
    }
    void releaseSubtensor(SubtensorDescriptor<int> &subtensor) DAAL_C11_OVERRIDE
    {
        releaseTSubtensor<int>(subtensor, "releaseIntSubtensor");
    }

    template<typename T>
    void getTSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, ReadWriteMode rwFlag,
        SubtensorDescriptor<T>& subtensor, const TensorOffsetLayout& layout , const char *javaMethodName, const char *javaMethodSignature)
    {
        jint status = JNI_OK;
        _tls local_tls = tls.local();

        size_t  nDim     = layout.getDimensions().size();
        const size_t* dimSizes = &((layout.getDimensions())[0]);
        size_t blockSize = subtensor.setDetails( nDim, dimSizes, fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwFlag );
        subtensor.saveOffsetLayout(layout);

        size_t* _dimOffsets = (size_t*)daal::services::daal_malloc(nDim * sizeof(size_t));

        if(!dimSizes) { this->_errors->add(services::ErrorNullParameterNotSupported); return; }

        for(size_t i=0; i<nDim; i++)
        {
            _dimOffsets[i] = 1;
        }

        for(size_t i=0; i<nDim; i++)
        {
            for(size_t j=0; j<i; j++)
            {
                _dimOffsets[j] *= dimSizes[i];
            }
        }

        size_t shift = 0;
        for( size_t i=0; i<fixedDims; i++ )
        {
            shift += fixedDimNums[i] * _dimOffsets[i];
        }
        if( fixedDims != nDim )
        {
            shift += rangeDimIdx * _dimOffsets[fixedDims];
        }

        /* Get JNI interface pointer for current thread */
        status = jvm->AttachCurrentThread((void **)(&(local_tls.jenv)), NULL);
        if(status != JNI_OK) { this->_errors->add(services::ErrorCouldntAttachCurrentThreadToJavaVM); return; }

        local_tls.is_attached = true;

        size_t bufferSize = blockSize * sizeof(T);

        if( !subtensor.resizeBuffer() ) { return; }

        void *buf = subtensor.getPtr();

        /* Get class associated with Java object */
        local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaTensor);
        if(local_tls.jcls == NULL)
        {
            this->_errors->add(services::ErrorCouldntFindClassForJavaObject);
            return;
        }

        /* Get ID of the 'getSubtensor' method of the Java class */
        jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, javaMethodName,
                                                        javaMethodSignature);
        if(jmeth == NULL)
        {
            services::SharedPtr<services::Error> e(new services::Error(services::ErrorCouldntFindJavaMethod));
            e->addStringDetail(services::Method, services::String(javaMethodName));
            this->_errors->add(e);
            return;
        }

        jlongArray jFixedDimNums = (local_tls.jenv)->NewLongArray(fixedDims);
        if (jFixedDimNums == NULL) {
            return; /* out of memory error thrown */
        }
        (local_tls.jenv)->SetLongArrayRegion(jFixedDimNums, 0, fixedDims, (jlong*)fixedDimNums);


        /* Call 'getSubtensor' Java method */
        local_tls.jbuf = (local_tls.jenv)->NewDirectByteBuffer( buf, bufferSize);
        local_tls.jbuf = (local_tls.jenv)->CallObjectMethod(jJavaTensor, jmeth, jFixedDimNums,
                                                            (jlong)rangeDimIdx, (jlong)rangeDimNum, local_tls.jbuf);

        buf = (local_tls.jenv)->GetDirectBufferAddress(local_tls.jbuf);

        tls.local() = local_tls;

        subtensor.setPtr( (T *)buf );

        if(!local_tls.is_main_thread)
        {
            status = jvm->DetachCurrentThread();
            if(status != JNI_OK)
            {
                this->_errors->add(services::ErrorCouldntAttachCurrentThreadToJavaVM);
                return;
            }
        }

        daal::services::daal_free( _dimOffsets );
    }

    template<typename T>
    void releaseTSubtensor(SubtensorDescriptor<T> &subtensor, const char *javaMethodName)
    {
        jint status = JNI_OK;
        _tls local_tls = tls.local();
        if(subtensor.getRWFlag() == writeOnly)
        {
            /* Get JNI interface pointer for current thread */
            status = jvm->AttachCurrentThread((void **)(&(local_tls.jenv)), NULL);
            if(status != JNI_OK)
            {
                this->_errors->add(services::ErrorCouldntAttachCurrentThreadToJavaVM);
                return;
            }

            /* Get class associated with Java object */
            local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaTensor);
            if(local_tls.jcls == NULL)
            {
                this->_errors->add(services::ErrorCouldntFindClassForJavaObject);
                return;
            }

            /* Get ID of the 'releaseSubtensor' method of the Java class */
            jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, javaMethodName, "([JJJLjava/nio/ByteBuffer;)V");
            if(jmeth == NULL)
            {
                services::SharedPtr<services::Error> e(new services::Error(services::ErrorCouldntFindJavaMethod));
                e->addStringDetail(services::Method, services::String(javaMethodName));
                this->_errors->add(e);
                return;
            }

            size_t nDim = getNumberOfDimensions();

            size_t blockSize = subtensor.getSize();
            const size_t* dimSizes = &((subtensor.getLayout()->getDimensions())[0]);

            size_t fixedDims     = subtensor.getFixedDims();
            size_t *fixedDimNums = subtensor.getFixedDimNums();
            size_t rangeDimIdx   = subtensor.getRangeDimIdx();
            size_t rangeDimNum   = subtensor.getRangeDimNum();

            size_t* _dimOffsets = (size_t*)daal::services::daal_malloc(nDim * sizeof(size_t));

            for(size_t i=0; i<nDim; i++)
            {
                _dimOffsets[i] = 1;
            }

            for(size_t i=0; i<nDim; i++)
            {
                for(size_t j=0; j<i; j++)
                {
                    _dimOffsets[j] *= dimSizes[i];
                }
            }

            size_t shift = 0;
            for( size_t i=0; i<fixedDims; i++ )
            {
                shift += fixedDimNums[i] * _dimOffsets[i];
            }
            if( fixedDims != nDim )
            {
                shift += rangeDimIdx * _dimOffsets[fixedDims];
            }

            jlongArray jFixedDimNums = (local_tls.jenv)->NewLongArray(fixedDims);
            if (jFixedDimNums == NULL) {
                return; /* out of memory error thrown */
            }
            (local_tls.jenv)->SetLongArrayRegion(jFixedDimNums, 0, fixedDims, (jlong*)fixedDimNums);

            size_t bufferSize = blockSize * sizeof(T);
            local_tls.jbuf = (local_tls.jenv)->NewDirectByteBuffer( subtensor.getPtr(), bufferSize );

            /* Call 'releaseSubtensor' Java method */
            (local_tls.jenv)->CallObjectMethod(
                jJavaTensor, jmeth, jFixedDimNums, (jlong)rangeDimIdx, (jlong)rangeDimNum, local_tls.jbuf);

            if(!local_tls.is_main_thread)
            {
                status = jvm->DetachCurrentThread();
                if(status != JNI_OK)
                {
                    this->_errors->add(services::ErrorCouldntAttachCurrentThreadToJavaVM);
                    return;
                }
            }

            daal::services::daal_free( _dimOffsets );
        }

        tls.local() = local_tls;
    }

    virtual void allocateDataMemory(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE {}

    virtual void freeDataMemory() DAAL_C11_OVERRIDE {}

    static void setJavaVM(JavaVM *jvm)
    {
        if (globalJavaVM == NULL)
        {
            globalJavaVM = jvm;
            Factory::instance().registerObject(new Creator<JavaTensor>());
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

    jobject getJavaObject() { return jJavaTensor; }

    virtual services::SharedPtr<Tensor> getSampleTensor(size_t firstDimIndex) DAAL_C11_OVERRIDE
    {
        return services::SharedPtr<Tensor>();
    }

    virtual bool check(services::ErrorCollection *errors, const char *description) const DAAL_C11_OVERRIDE
    {
        return true;
    }

    virtual TensorOffsetLayout createDefaultSubtensorLayout() const DAAL_C11_OVERRIDE
    {
        return TensorOffsetLayout(_layout);
    }

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
    jobject jJavaTensor;                      /**< Java object associated with this C++ object */
    JavaVM *jvm;                                /**< Java VM interface function table */

private:
    static JavaVM *globalJavaVM;
    static tbb::enumerable_thread_specific<jobject> globalDaalContext;
    TensorOffsetLayout _layout;
};

} // namespace daal

#endif
