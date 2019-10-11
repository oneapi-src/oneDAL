/* file: xavier_initializer.h */
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
//  Implementation of the Xavier initializer in the batch processing mode
//--
*/

#ifndef __XAVIER_INITIALIZER_H__
#define __XAVIER_INITIALIZER_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/initializers/initializer.h"
#include "algorithms/neural_networks/initializers/initializer_types.h"
#include "algorithms/neural_networks/initializers/xavier/xavier_initializer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace xavier
{
/**
 * @defgroup initializers_xavier_batch Batch
 * @ingroup initializers_xavier
 * @{
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__XAVIER__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the Xavier initializer.
 *        This class is associated with the \ref xavier::interface1::Batch "xavier::Batch" class
 *        and supports the method of Xavier initializer computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of Xavier initializer, double or float
 * \tparam method           Computation method of the initializer, xavier::Method
 * \tparam cpu              Version of the cpu-specific implementation of the initializer, daal::CpuType
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public initializers::InitializerContainerIface
{
public:
    /**
     * Constructs a container for the Xavier initializer with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the Xavier initializer in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__XAVIER__BATCH"></a>
 * \brief Provides methods for Xavier initializer computations in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of Xavier initializer, double or float
 * \tparam method           Computation method of the initializer, Xavier::Method
 *
 * \par Enumerations
 *      - xavier::Method          Computation methods for the Xavier initializer
 *
 * \par References
 *      - \ref initializers::interface1::Input "initializers::Input" class
 *      - \ref initializers::interface1::Result "initializers::Result" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public initializers::InitializerIface
{
public:
    typedef initializers::InitializerIface super;

    typedef typename super::InputType                                    InputType;
    typedef algorithms::neural_networks::initializers::xavier::Parameter ParameterType;
    typedef typename super::ResultType                                   ResultType;

    ParameterType parameter; /*!< %Parameters of the initializer */

    /**
     * Constructs Xavier initializer
     */
    Batch()
    {
        initialize();
    }

    /**
     * Constructs Xavier initializer by copying input objects and parameters of another Xavier initializer
     * \param[in] other An initializer to be used as the source to initialize the input objects
     *                  and parameters of this initializer
     */
    Batch(const Batch<algorithmFPType, method> &other) : super(other), parameter(other.parameter)
    {
        initialize();
    }

    /**
     * Returns method of the initializer
     * \return Method of the initializer
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Get parameters of the initializer
     * \return Parameters of the initializer
     */
    virtual ParameterType * getParameter() DAAL_C11_OVERRIDE { return &parameter; }

    /**
     * Returns the structure that contains results of Xavier initializer
     * \return Structure that contains results of Xavier initializer
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store results of Xavier initializer
     * \param[in] result  Structure to store results of Xavier initializer
     *
     * \return Status of computations
     */
    services::Status setResult(const ResultPtr &result)
    {
        _result = result;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated Xavier initializer
     * with a copy of input objects and parameters of this Xavier initializer
     * \return Pointer to the newly allocated initializer
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
     * Allocates memory to store the result of the Xavier initializer
     *
     * \return Status of computations
     */
    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        _par = &parameter;
        services::Status s = this->_result->template allocate<algorithmFPType>(&(this->input), &parameter, (int) method);
        this->_res = this->_result.get();
        return s;
    }

protected:
    virtual Batch<algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }

private:
    ResultPtr _result;
};

} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;
/** @} */
} // namespace xavier
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
