/* file: java_batch.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
class JavaBatch : public BatchImpl
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

    virtual services::Status setResult(const ResultPtr &result) DAAL_C11_OVERRIDE
    {
        _result = result;
        (static_cast<JavaBatchContainer*>(this->_ac))->setJavaResult(_result);
        _res = _result.get();
        return services::Status();
    }

protected:
    virtual services::Status allocateResult() DAAL_C11_OVERRIDE // To make the class non-abstract
    {
        services::Status s = _result->allocate<double>(_in, (daal::algorithms::Parameter *) (&parameter), 0);
        _res = _result.get();
        return services::Status();
    }

    virtual JavaBatch * cloneImpl() const DAAL_C11_OVERRIDE { return NULL; }
};

} // namespace daal::algorithms::covariance
} // namespace daal::algorithms
} // namespace daal

#endif
