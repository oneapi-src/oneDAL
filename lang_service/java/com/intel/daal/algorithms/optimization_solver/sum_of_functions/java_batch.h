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
//  Implementation of the class that connects Java Batch
//  to C++ algorithm
//--
*/
#ifndef __JAVA_BATCH_H__
#define __JAVA_BATCH_H__

#include <jni.h>
#include <tbb/tbb.h>

#include "algorithms/optimization_solver/objective_function/sum_of_functions_types.h"
#include "algorithms/optimization_solver/objective_function/sum_of_functions_batch.h"
#include "java_callback.h"
#include "java_batch_container.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{

using namespace daal::data_management;
using namespace daal::services;

namespace sum_of_functions
{

/*
 * \brief Class that specifies the default method for partial results initialization
 */
class JavaBatch : public sum_of_functions::Batch
{
public:
    /** Default constructor */
    JavaBatch(size_t numberOfTerms, JavaVM *_jvm, jobject _javaObject) : sum_of_functions::Batch(numberOfTerms, 0, 0)
    {
        JavaBatchContainer *_container = new JavaBatchContainer(_jvm, _javaObject);
        _container->setJavaResult(_result);
        _container->setEnvironment(&_env);

        this->_ac = _container;
    }

    JavaBatch(const JavaBatch &other) :
        sum_of_functions::Batch(other.sumOfFunctionsParameter->numberOfTerms, 0, 0)
    {
        JavaBatchContainer *_container = (static_cast<JavaBatchContainer *>(other._ac))->cloneImpl();
        sumOfFunctionsParameter = static_cast<sum_of_functions::Parameter *>(_container->_parameter);
        sumOfFunctionsInput = static_cast<sum_of_functions::Input *>(_container->_input);
        this->_par = sumOfFunctionsParameter;
        this->_in = sumOfFunctionsInput;
        _container->setJavaResult(_result);

        this->_ac = _container;
    }

    void setPointersToContainer(sum_of_functions::Input *ptrInput, sum_of_functions::Parameter *ptrParameter)
    {
        sumOfFunctionsParameter = ptrParameter;
        sumOfFunctionsInput = ptrInput;
        _in = ptrInput;
        _par = ptrParameter;
    }

    virtual ~JavaBatch() {}

    virtual int getMethod() const DAAL_C11_OVERRIDE { return 0; } // To make the class non-abstract

    virtual services::Status setResult(const objective_function::ResultPtr &result) DAAL_C11_OVERRIDE
    {
        _result = result;
        (static_cast<JavaBatchContainer *>(this->_ac))->setJavaResult(_result);
        _res = _result.get();
        return services::Status();
    }

protected:
    virtual services::Status allocateResult() DAAL_C11_OVERRIDE // To make the class non-abstract
    {
        services::Status s = _result->allocate<double>(_in, _par, 0);
        _res = _result.get();
        return s;
    }

    virtual JavaBatch *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new JavaBatch(*this);
    }
};

} // namespace daal::algorithms::optimization_solver::sum_of_functions
} // namespace daal::algorithms::optimization_solver
} // namespace daal::algorithms
} // namespace daal

#endif
