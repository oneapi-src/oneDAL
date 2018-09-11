/* file: truncated_gaussian_initializer.h */
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
//  Implementation of the truncated gaussian initializer in the batch processing mode
//--
*/

#ifndef __TRUNCATED_GAUSSIAN_INITIALIZER_H__
#define __TRUNCATED_GAUSSIAN_INITIALIZER_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/initializers/initializer.h"
#include "algorithms/neural_networks/initializers/initializer_types.h"
#include "algorithms/neural_networks/initializers/truncated_gaussian/truncated_gaussian_initializer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace truncated_gaussian
{
/**
 * @defgroup initializers_truncated_gaussian_batch Batch
 * @ingroup initializers_truncated_gaussian
 * @{
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__TRUNCATED_GAUSSIAN__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the truncated gaussian initializer.
 *        This class is associated with the \ref truncated_gaussian::interface1::Batch "truncated_gaussian::Batch" class
 *        and supports the method of truncated gaussian initializer computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of truncated gaussian initializer, double or float
 * \tparam method           Computation method of the initializer, truncated_gaussian::Method
 * \tparam cpu              Version of the cpu-specific implementation of the initializer, daal::CpuType
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public initializers::InitializerContainerIface
{
public:
    /**
     * Constructs a container for the truncated gaussian initializer with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the truncated gaussian initializer in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__TRUNCATED_GAUSSIAN__BATCH"></a>
 * \brief Provides methods for truncated gaussian initializer computations in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of truncated gaussian initializer, double or float
 * \tparam method           Computation method of the initializer, truncated_gaussian::Method
 *
 * \par Enumerations
 *      - truncated_gaussian::Method          Computation methods for the truncated gaussian initializer
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

    typedef typename super::InputType                                                                 InputType;
    typedef algorithms::neural_networks::initializers::truncated_gaussian::Parameter<algorithmFPType> ParameterType;
    typedef typename super::ResultType                                                                ResultType;

    Parameter<algorithmFPType> parameter; /*!< %Parameters of the initializer */

    /**
     * Constructs truncated gaussian initializer
     *  \param[in] mean   Mean
     *  \param[in] sigma  Standard deviation
     */
    Batch(double mean = 0, double sigma = 1.00) : parameter(mean, sigma)
    {
        initialize();
    }

    /**
     * Constructs truncated gaussian initializer by copying input objects and parameters of another truncated gaussian initializer
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
     * Returns the structure that contains results of truncated gaussian initializer
     * \return Structure that contains results of truncated gaussian initializer
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store results of truncated gaussian initializer
     * \param[in] result  Structure to store results of truncated gaussian initializer
     *
     * \return Status of computations
     */
    services::Status setResult(const ResultPtr &result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated truncated gaussian initializer
     * with a copy of input objects and parameters of this truncated gaussian initializer
     * \return Pointer to the newly allocated initializer
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
     * Allocates memory to store the result of the truncated gaussian initializer
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
} // namespace truncated_gaussian
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
