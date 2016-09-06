/* file: java_batch.h */
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
//  Implementation of the class that connects Covariance Java Batch
//  to C++ algorithm
//--
*/
#ifndef __JAVA_BATCH_H__
#define __JAVA_BATCH_H__

#include <jni.h>
#include <tbb/tbb.h>

#include "algorithms/covariance/covariance_types.h"
#include "algorithms/covariance/covariance_batch.h"
#include "java_callback.h"
#include "java_batch_container.h"

namespace daal
{
namespace algorithms
{
namespace covariance
{

using namespace daal::data_management;
using namespace daal::services;

/*
 * \brief Class that specifies the default method for partial results initialization
 */
class JavaBatch : public BatchIface
{
public:
    /** Default constructor */
    JavaBatch(JavaVM *_jvm, jobject _javaObject)
    {
        JavaBatchContainer* _container = new JavaBatchContainer(_jvm, _javaObject);
        _container->setJavaResult(_result);
        _container->setEnvironment(&_env);

        this->_ac = _container ;
    };

    virtual ~JavaBatch() {}

    virtual int getMethod() const DAAL_C11_OVERRIDE { return 0; } // To make the class non-abstract

    virtual void setResult(const services::SharedPtr<Result> &result) DAAL_C11_OVERRIDE
    {
        _result = result;
        (static_cast<JavaBatchContainer*>(this->_ac))->setJavaResult(_result);
        _res = _result.get();
    }

protected:
    virtual void allocateResult() DAAL_C11_OVERRIDE // To make the class non-abstract
    {
        _result->allocate<double>(_in, (daal::algorithms::Parameter *) (&parameter), 0);
        _res = _result.get();
    }

    virtual JavaBatch * cloneImpl() const DAAL_C11_OVERRIDE { return NULL; }
};

} // namespace daal::algorithms::covariance
} // namespace daal::algorithms
} // namespace daal

#endif
