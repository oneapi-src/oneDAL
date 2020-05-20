/* file: kmeans_distributed_batch.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  Implementation of the interface for K-Means algorithm in the batch
//  processing mode
//--
*/

#ifndef __KMEANS_DISTRIBUTED_BATCH_H__
#define __KMEANS_DISTRIBUTED_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/kmeans/kmeans_types.h"

namespace daal
{
namespace algorithms
{
namespace preview
{
namespace distributed_kmeans
{
namespace interface1
{
enum Method
{
    defaultDense = 0
};
/**
 * @defgroup kmeans_batch Batch
 * @ingroup kmeans_compute
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTRIBUTED_BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of K-Means algorithm.
 *        This class is associated with the daal::algorithms::preview::distributed_kmeans::Batch class
 *        and supports the method of K-Means computation in the batch processing mode
 *        but using multi-process communication internally.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of K-Means, double or float
  */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for K-Means algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of K-Means algorithm in the batch processing mode
     */
    virtual daal::services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__BATCH"></a>
 * \brief Computes the results of K-Means algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of K-Means, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for K-Means algorithm
 *      - \ref InputId  Identifiers of input objects for K-Means algorithm
 *      - \ref ResultId Identifiers of results of K-Means algorithm
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::kmeans::Input InputType;
    typedef algorithms::kmeans::Parameter ParameterType;
    typedef algorithms::kmeans::Result ResultType;

    /**
     *  Main constructor
     *  \param[in] nClusters   Number of clusters
     *  \param[in] nIterations Number of iterations
     */
    Batch(size_t nClusters, size_t nIterations = 1) : parameter(nClusters, nIterations) { initialize(); }

    /**
     * Constructs K-Means algorithm by copying input objects and parameters
     * of another K-Means algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other) : parameter(other.parameter)
    {
        initialize();
        input.set(algorithms::kmeans::data, other.input.get(algorithms::kmeans::data));
        input.set(algorithms::kmeans::inputCentroids, other.input.get(algorithms::kmeans::inputCentroids));
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains the results of K-Means algorithm
     * \return Structure that contains the results of K-Means algorithm
     */
    algorithms::ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated  memory  to store the results of K-Means algorithm
     * \param[in] result  Structure to store the results of K-Means algorithm
     */
    daal::services::Status setResult(const algorithms::ResultPtr & result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res    = _result.get();
        return daal::services::Status();
    }

    /**
     * Returns a pointer to the newly allocated K-Means algorithm with a copy of input objects
     * and parameters of this K-Means algorithm
     * \return Pointer to the newly allocated algorithm
     */
    daal::services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return daal::services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    virtual daal::services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        daal::services::Status s = ((algorithms::kmeans::Result *)_result.get())->allocate<algorithmFPType>(_in, _par, (int)method);
        _res                     = _result.get();
        return s;
    }

    void initialize()
    {
        using daal::algorithms::interface1::AlgorithmDispatchContainer;
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in                  = &input;
        _par                 = &parameter;
    }

public:
    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< K-Means parameters structure */

private:
    algorithms::ResultPtr _result;

    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace distributed_kmeans
} // namespace preview
} // namespace algorithms
} // namespace daal
#endif
