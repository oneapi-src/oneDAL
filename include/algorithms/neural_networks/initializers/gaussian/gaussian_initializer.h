/* file: gaussian_initializer.h */
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
//  Implementation of the gaussian initializer in the batch processing mode
//--
*/

#ifndef __GAUSSIAN_INITIALIZER_H__
#define __GAUSSIAN_INITIALIZER_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/initializers/initializer.h"
#include "algorithms/neural_networks/initializers/initializer_types.h"
#include "algorithms/neural_networks/initializers/gaussian/gaussian_initializer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace gaussian
{
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__GAUSSIAN__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the gaussian initializer.
 *        This class is associated with the \ref gaussian::interface1::Batch "gaussian::Batch" class
 *        and supports the method of gaussian initializer computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of gaussian initializer, double or float
 * \tparam method           Computation method of the initializer, gaussian::Method
 * \tparam cpu              Version of the cpu-specific implementation of the initializer, daal::CpuType
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the gaussian initializer with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the gaussian initializer in the batch processing mode
     */
    void compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__GAUSSIAN__BATCH"></a>
 * \brief Provides methods for gaussian initializer computations in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of gaussian initializer, double or float
 * \tparam method           Computation method of the initializer, gaussian::Method
 *
 * \par Enumerations
 *      - gaussian::Method          Computation methods for the gaussian initializer
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - \ref initializers::interface1::Input "initializers::Input" class
 *      - \ref initializers::interface1::Result "initializers::Result" class
 */
template<typename algorithmFPType = float, Method method = defaultDense>
class DAAL_EXPORT Batch : public initializers::InitializerIface
{
public:
    Parameter parameter; /*!< %Parameters of the initializer */

    /**
     * Constructs gaussian initializer
     *  \param[in] a     Mean
     *  \param[in] sigma Standard deviation
     *  \param[in] seed  Seed for generating random numbers for the initialization
     */
    Batch(double a = 0, double sigma = 0.01, size_t seed = 777);

    /**
     * Constructs gaussian initializer by copying input objects and parameters of another gaussian initializer
     * \param[in] other An initializer to be used as the source to initialize the input objects
     *                  and parameters of this initializer
     */
    Batch(const Batch<algorithmFPType, method> &other): parameter(other.parameter)
    {
        initialize();
        input.set(initializers::data, other.input.get(initializers::data));
    }

    /**
     * Returns method of the initializer
     * \return Method of the initializer
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains results of gaussian initializer
     * \return Structure that contains results of gaussian initializer
     */
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store results of gaussian initializer
     * \param[in] result  Structure to store results of gaussian initializer
     */
    void setResult(const services::SharedPtr<Result> &result)
    {
        _result = result;
        _res = _result.get();
    }

    /**
     * Returns a pointer to the newly allocated gaussian initializer
     * with a copy of input objects and parameters of this gaussian initializer
     * \return Pointer to the newly allocated initializer
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
     * Allocates memory to store the result of the gaussian initializer
     */
    virtual void allocateResult() DAAL_C11_OVERRIDE
    {
        _par = &parameter;
        this->_result->template allocate<algorithmFPType>(&(this->input), &parameter, (int) method);
        this->_res = this->_result.get();
    }

protected:
    virtual Batch<algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    void initialize()
    {
        parameterBase = &parameter;
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _result = services::SharedPtr<Result>(new Result());
    }

private:
    services::SharedPtr<Result> _result;
};

} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;
} // namespace gaussian
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
