/* file: kmeans_batch.h */
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
//  Implementation of the interface for the K-Means algorithm in the batch
//  processing mode
//--
*/

#ifndef __KMEANS_BATCH_H__
#define __KMEANS_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/kmeans/kmeans_types.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{

namespace interface1
{
/**
 * @defgroup kmeans_batch Batch
 * @ingroup kmeans_compute
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the K-Means algorithm.
 *        This class is associated with the daal::algorithms::kmeans::Batch class
 *        and supports the method of K-Means computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of K-Means, double or float
 * \tparam method           Computation method of the algorithm, \ref daal::algorithms::kmeans::Method
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the K-Means algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of the K-Means algorithm in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__BATCH"></a>
 * \brief Computes the results of the K-Means algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of K-Means, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the K-Means algorithm
 *      - \ref InputId  Identifiers of input objects for the K-Means algorithm
 *      - \ref ResultId Identifiers of results of the K-Means algorithm
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = lloydDense>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::kmeans::Input     InputType;
    typedef algorithms::kmeans::Parameter ParameterType;
    typedef algorithms::kmeans::Result    ResultType;

    /**
     *  Main constructor
     *  \param[in] nClusters   Number of clusters
     *  \param[in] nIterations Number of iterations
     */
    Batch(size_t nClusters, size_t nIterations = 1) : parameter(nClusters, nIterations)
    {
        initialize();
    }

    /**
     * Constructs a K-Means algorithm by copying input objects and parameters
     * of another K-Means algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : parameter(other.parameter)
    {
        initialize();
        input.set(data, other.input.get(data));
        input.set(inputCentroids, other.input.get(inputCentroids));
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains the results of the K-Means algorithm
     * \return Structure that contains the results of the K-Means algorithm
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated  memory  to store the results of the K-Means algorithm
     * \param[in] result  Structure to store the results of the K-Means algorithm
     */
    services::Status setResult(const ResultPtr& result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated K-Means algorithm with a copy of input objects
     * and parameters of this K-Means algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        services::Status s = _result->allocate<algorithmFPType>(_in, _par, (int) method);
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
    InputType input;            /*!< %Input data structure */
    ParameterType parameter;    /*!< K-Means parameters structure */

private:
    ResultPtr _result;
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
#endif
