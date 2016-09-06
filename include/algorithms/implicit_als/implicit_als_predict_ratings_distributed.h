/* file: implicit_als_predict_ratings_distributed.h */
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
//  Implementation of the interface for implicit ALS model-based ratings prediction
//  in the distributed processing mode
//--
*/

#ifndef __IMPLICIT_ALS_PREDICT_RATINGS_DISTRIBUTED_H__
#define __IMPLICIT_ALS_PREDICT_RATINGS_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "algorithms/implicit_als/implicit_als_predict_ratings_types.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace prediction
{
namespace ratings
{

namespace interface1
{
/**
 * @defgroup implicit_als_prediction_distributed Distributed
 * @ingroup implicit_als_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__DISTRIBUTEDCONTAINER"></a>
 * \brief Class that contains methods to run implicit ALS model-based prediction in the distributed processing mode
 */
template<ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class that contains methods to run implicit ALS model-based prediction in the first step of
 *        the distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step1Local, algorithmFPType, method, cpu> : public DistributedPredictionContainerIface
{
public:
     /**
     * Constructs a container for implicit ALS model-based ratings prediction with a specified environment
     * in the first step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
    ~DistributedContainer();

    /**
     * Computes a partial result of implicit ALS model-based prediction
     * in the first step of the distributed processing mode
     */
    void compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of implicit ALS model-based prediction
     * in the first step of the distributed processing mode
     */
    void finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__DISTRIBUTED"></a>
 * \brief Runs implicit ALS model-based prediction in the distributed processing mode
 * \n<a href="DAAL-REF-IMPLICIT_ALS-ALGORITHM">Implicit ALS algorithm description and usage models</a>
 *
 * \tparam step             Step of the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for implicit ALS model-based prediction, double or float
 * \tparam method           Implicit ALS prediction method, \ref Method
 *
 * \par Enumerations
 *      - \ref ComputeStep  Computation steps
 *      - \ref Method       Computation methods
 *
 * \par References
 *      - \ref implicit_als::interface1::Parameter "implicit_als::Parameter" class
 *      - \ref Distributed class
 */
template<ComputeStep step, typename algorithmFPType = double, Method method = defaultDense>
class Distributed : public daal::algorithms::DistributedPrediction {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Performs implicit ALS model-based prediction in the first step of the distributed processing mode
 * <a href="DAAL-REF-IMPLICIT_ALS-ALGORITHM">Implicit ALS algorithm description and usage models</a>
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for implicit ALS model-based prediction, double or float
 * \tparam method           Implicit ALS prediction method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref implicit_als::interface1::Parameter "implicit_als::Parameter" class
 *      - \ref DistributedInput<step1Local> class
 *      - \ref PartialResult class
 *      - \ref Result class
 */
template<typename algorithmFPType, Method method>
class Distributed<step1Local, algorithmFPType, method> : public daal::algorithms::DistributedPrediction
{
public:
    DistributedInput<step1Local> input;             /*!< %Input data structure */
    Parameter parameter;                            /*!< Parameters of the algorithm */

    /**
     * Default constructor
     */
    Distributed()
    {
        initialize();
    }

    /**
     * Constructs an implicit ALS ratings prediction algorithm by copying input objects and parameters
     * of another implicit ALS ratings prediction algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step1Local, algorithmFPType, method> &other)
    {
        initialize();
        input.set(usersPartialModel, other.input.get(usersPartialModel));
        input.set(itemsPartialModel, other.input.get(itemsPartialModel));
        parameter = other.parameter;
    }

    virtual ~Distributed() {}

    /**
     * Returns the structure that contains the results of the implicit ALS ratings prediction algorithm
     * \return Structure that contains the results of the implicit ALS ratings prediction algorithm
     */
    services::SharedPtr<Result> getResult()
    {
        return _partialResult->get(finalResult);
    }

    /**
     * Returns the structure that contains computed partial results of the implicit ALS ratings prediction algorithm
     * \return Structure that contains computed partial results of the implicit ALS ratings prediction algorithm
     */
    services::SharedPtr<PartialResult> getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Registers user-allocated memory for storing the prediction results
     * \param[in] result Structure for storing the prediction results
     */
    void setResult(const services::SharedPtr<Result> &result)
    {
        DAAL_CHECK(result, ErrorNullResult)
        _partialResult->set(finalResult, result);
        _pres = _partialResult.get();
    }

    /**
     * Registers user-allocated memory for storing partial results of the implicit ALS ratings prediction algorithm
     * \param[in] partialResult  Structure for storing partial results of the implicit ALS ratings prediction algorithm
     * \param[in] initFlag       Flag that specifies whether partial results are initialized
     */
    void setPartialResult(const services::SharedPtr<PartialResult>& partialResult, bool initFlag = false)
    {
        DAAL_CHECK(partialResult->get(finalResult), ErrorNullResult)
        _partialResult = partialResult;
        _pres = _partialResult.get();
        setInitFlag(initFlag);
    }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns a pointer to the newly allocated ALS ratings prediction algorithm with a copy of input objects
     * and parameters of this ALS ratings prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step1Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step1Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    services::SharedPtr<PartialResult> _partialResult;

    virtual Distributed<step1Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step1Local, algorithmFPType, method>(*this);
    }

    virtual void allocateResult() DAAL_C11_OVERRIDE {}

    virtual void allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult->allocate<algorithmFPType>(&input, &parameter, (int)method);
        _pres = _partialResult.get();
    }

    virtual void initializePartialResult() DAAL_C11_OVERRIDE {}

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step1Local, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _partialResult = services::SharedPtr<PartialResult>(new PartialResult());
    }

};
/** @} */
} // interface1
using interface1::DistributedContainer;
using interface1::Distributed;

}
}
}
}
}
#endif
