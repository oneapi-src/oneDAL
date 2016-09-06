/* file: em_gmm.h */
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
//  Implementation of the interface for the EM for GMM algorithm
//--
*/

#ifndef __EM_GMM_H__
#define __EM_GMM_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "em_gmm_types.h"

namespace daal
{
namespace algorithms
{
namespace em_gmm
{

namespace interface1
{
/** @defgroup em_gmm_batch Batch
 * @ingroup em_gmm_compute
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the EM for GMM algorithm.
 *        This class is associated with the Batch class and supports the method of computing EM for GMM in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the EM for GMM algorithm, double or float
 * \tparam method           EM for GMM computation method
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
     /**
     * Constructs a container for the EM for GMM algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the EM for GMM algorithm in the batch processing mode
     */
    virtual void compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__BATCH"></a>
 * \brief Computes EM for GMM in the batch processing mode.
 * \n<a href="DAAL-REF-EM_GMM-ALGORITHM">EM for GMM algorithm description and usage models</a>
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the EM for GMM algorithm, double or float
 * \tparam method           EM for GMM computation method
 *
 * \par Enumerations
 *      - \ref Method Computation methods for EM for GMM
 *      - \ref InputId  Identifiers of input objects for EM for GMM
 *      - \ref ResultId Result identifiers for EM for GMM
 *
 * \par References
 *      - Input class
 *      - Result class
 */
template<typename algorithmFPType = double, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    Batch(const size_t nComponents);

    /**
     * Constructs an EM for GMM algorithm by copying input objects and parameters
     * of another EM for GMM algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : parameter(other.parameter)
    {
        initialize();
        input.set(data,             other.input.get(data));
        input.set(inputWeights,     other.input.get(inputWeights));
        input.set(inputMeans,       other.input.get(inputMeans));
        input.set(inputCovariances, other.input.get(inputCovariances));
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns the structure that contains results of the EM for GMM algorithm
     * \return Structure that contains results of the EM for GMM algorithm
     */
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Sets the memory for storing results of the EM for GMM algorithm
     * \param[in] result  Structure for storing results of the EM for GMM algorithm
     */
    void setResult(const services::SharedPtr<Result>& result)
    {
        DAAL_CHECK(result, ErrorNullResult)
        _result = result;
        _res = _result.get();
    }

    /**
     * Returns a pointer to the newly allocated EM for GMM algorithm with a copy of input objects
     * of this EM for GMM algorithm
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

    virtual void allocateResult() DAAL_C11_OVERRIDE
    {
        _result->allocate<algorithmFPType>(&input, &parameter, (int) method);
        _res = _result.get();
    }

    void initialize();

public:
    Input input;           /*!< %Input data structure */
    Parameter parameter;   /*!< %Parameter data structure */

private:
    services::SharedPtr<Result> _result;
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace em_gmm
} // namespace algorithm
} // namespace daal
#endif
