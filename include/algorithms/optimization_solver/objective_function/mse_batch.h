/* file: mse_batch.h */
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
//  Implementation of the mean squared error (MSE) objective function in the batch
//  processing mode
//--
*/

#ifndef __MSE_BATCH_H__
#define __MSE_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "sum_of_functions_batch.h"
#include "mse_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace mse
{

namespace interface1
{
/** @defgroup mse_batch Batch
 * @ingroup mse
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the mean squared error objective function.
 *        This class is associated with the Batch class and supports the method of computing
 *        the Mean squared error objective function in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the Mean squared error objective function, double or float
 * \tparam method           the mean squared error objective function computation method
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the MSE objective function with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of the MSE objective function in the batch processing mode
     */
    virtual void compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__BATCH"></a>
 * \brief Computes the Mean squared error objective function in the batch processing mode.
 * \n<a href="DAAL-REF-MSE-ALGORITHM">The Mean squared error objective function algorithm description and usage models</a>
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the Mean squared error objective function, double or float
 * \tparam method           The Mean squared error objective function computation method
 *
 * \par Enumerations
 *      - Method Computation methods for the Mean squared error objective function
 *      - InputId  Identifiers of input objects for the Mean squared error objective function
 *      - ResultId Result identifiers for the Mean squared error objective function
 *
 * \par References
 *      - Input class
 *      - Result class
 */
template<typename algorithmFPType = double, Method method = defaultDense>
class DAAL_EXPORT Batch : public sum_of_functions::Batch
{
public:
    /**
     *  Main constructor
     */
    Batch(size_t numberOfTerms) : parameter(numberOfTerms), sum_of_functions::Batch(numberOfTerms, &input, &parameter)
    {
        initialize();
    }

    virtual ~Batch() {}

    /**
     * Constructs an the Mean squared error objective function algorithm by copying input objects and parameters
     * of another the Mean squared error objective function algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) :
        parameter(other.parameter), sum_of_functions::Batch(other.parameter.numberOfTerms, &input, &parameter)
    {
        initialize();
        input.set(argument, other.input.get(argument));
        input.set(data, other.input.get(data));
        input.set(dependentVariables, other.input.get(dependentVariables));
    }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns a pointer to the newly allocated the Mean squared error objective function algorithm with a copy of input objects
     * of this the Mean squared error objective function algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
     * Allocates memory buffers needed for the computations
     */
    void allocate()
    {
        allocateResult();
    }

protected:
    virtual Batch<algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    virtual void allocateResult() DAAL_C11_OVERRIDE
    {
        _result->allocate<algorithmFPType>(&input, &parameter, (int) method);
        _res = _result.get();
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
    }

public:
    Input input;           /*!< %Input data structure */
    Parameter parameter;   /*!< %Parameter data structure */

};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace mse
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
