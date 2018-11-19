/* file: mse_batch.h */
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
/**
 * @defgroup mse_batch Batch
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
     *
     * \return Status of computations
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__BATCH"></a>
 * \brief Computes the Mean squared error objective function in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-MSE-ALGORITHM">The Mean squared error objective function algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the Mean squared error objective function, double or float
 * \tparam method           The Mean squared error objective function computation method
 *
 * \par Enumerations
 *      - \ref Method Computation methods for the Mean squared error objective function
 *      - \ref InputId  Identifiers of input objects for the Mean squared error objective function
 *      - \ref objective_function::ResultId %Result identifiers for the Mean squared error objective function
 *
 * \par References
 *      - \ref objective_function::interface1::Result "Result" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public sum_of_functions::Batch
{
public:
    typedef sum_of_functions::Batch super;

    typedef algorithms::optimization_solver::mse::Input     InputType;
    typedef algorithms::optimization_solver::mse::Parameter ParameterType;
    typedef typename super::ResultType                      ResultType;

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
        parameter(other.parameter), sum_of_functions::Batch(other.parameter.numberOfTerms, &input, &parameter), input(other.input)
    {
        initialize();
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
     *
     * \return Status of computations
     */
    services::Status allocate()
    {
        return allocateResult();
    }

protected:
    virtual Batch<algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, &parameter, (int) method);
        _res = _result.get();
        return s;
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
    }

public:
    InputType input;           /*!< %Input data structure */
    ParameterType parameter;   /*!< %Parameter data structure */

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
