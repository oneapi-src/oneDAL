/* file: java_distributed.h */
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
//  Implementation of the class that connects Covariance Java Distributed
//  to C++ algorithm
//--
*/
#ifndef __JAVA_DISTRIBUTED_H__
#define __JAVA_DISTRIBUTED_H__

#include <jni.h>

#include "algorithms/covariance/covariance_types.h"
#include "algorithms/covariance/covariance_distributed.h"
#include "java_callback.h"
#include "java_distributed_container.h"

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
class JavaDistributed : public DistributedIface<step2Master>
{
public:
    /** Default constructor */
    JavaDistributed(JavaVM *_jvm, jobject _javaObject)
    {
        JavaDistributedContainer* _container = new JavaDistributedContainer(_jvm, _javaObject);
        _container->setJavaResult(_result);
        _container->setJavaPartialResult(_partialResult);

        _container->setEnvironment(&_env);

        this->_ac = _container;
    };

    virtual ~JavaDistributed() {}

    virtual int getMethod() const DAAL_C11_OVERRIDE { return 0; } // To make the class non-abstract

    virtual services::Status setResult(const ResultPtr &result) DAAL_C11_OVERRIDE
    {
        _result = result;
        (static_cast<JavaDistributedContainer*>(this->_ac))->setJavaResult(_result);
        _res = _result.get();
        return services::Status();
    }

    virtual services::Status setPartialResult(const PartialResultPtr &partialResult, bool _initFlag = false) DAAL_C11_OVERRIDE
    {
        _partialResult = partialResult;
        (static_cast<JavaDistributedContainer*>(this->_ac))->setJavaPartialResult(_partialResult);
        _pres = _partialResult.get();
        return services::Status();
    }

protected:
    virtual JavaDistributed * cloneImpl() const DAAL_C11_OVERRIDE { return NULL; }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<double>(_partialResult.get(), _par, 0);
        _res    = _result.get();
        _pres   = _partialResult.get();
        return s;
    }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->allocate<double>(&input, _par, 0);
        _pres   = _partialResult.get();
        return s;
    }
};

} // namespace daal::algorithms::covariance
} // namespace daal::algorithms
} // namespace daal

#endif
