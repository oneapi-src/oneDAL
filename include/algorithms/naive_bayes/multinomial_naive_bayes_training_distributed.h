/* file: multinomial_naive_bayes_training_distributed.h */
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
//  Implementation of the interface for multinomial naive Bayes model-based training
//  in the distributed processing mode
//--
*/

#ifndef __NAIVE_BAYES_TRAINING_DISTRIBUTED_H__
#define __NAIVE_BAYES_TRAINING_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "multinomial_naive_bayes_training_types.h"
#include "algorithms/classifier/classifier_training_distributed.h"

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace training
{

namespace interface1
{
/**
 * @defgroup multinomial_naive_bayes_training_distributed Distributed
 * @ingroup multinomial_naive_bayes_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__DISTRIBUTEDCONTAINER"></a>
 *  \brief Class containing methods to compute naive Bayes training results in the distributed processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the naive Bayes training algorithm in the distributed processing mode,
 *                          double or float
 * \tparam method           Naive Bayes training method on the first step in distributed processing mode, \ref Method
 */
template<ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods to train naive Bayes model in the distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step1Local, algorithmFPType, method, cpu> : public OnlineContainer<algorithmFPType, method, cpu> {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__DISTRIBUTEDCONTAINER_STEP2MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods to train naive Bayes in the distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step2Master, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for multinomial naive Bayes model-based training with a specified environment
     * in the second step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of naive Bayes model-based training
     * in the second step of the distributed processing mode
     */
    void compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of naive Bayes model-based training
     * in the second step of the distributed processing mode
     */
    void finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__DISTRIBUTED_STEP_ALGORITHMFPTTYPE_METHOD"></a>
 *  \brief Algorithm class for training naive Bayes model in the distributed processing mode
 *  \n<a href="DAAL-REF-MULTINOMNAIVEBAYES-ALGORITHM">Multinomial naive Bayes algorithm description and usage models</a>
 *
 *  \tparam algorithmFPType  Data type to use in intermediate computations for multinomial naive Bayes training, double or float
 *  \tparam method           Computation method, \ref Method
 *
 *  \par Enumerations
 *      - \ref Method Training methods for the naive Bayes algorithm
 *
 */
template<ComputeStep step, typename algorithmFPType = double, Method method = defaultDense>
class DAAL_EXPORT Distributed {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTTYPE_METHOD"></a>
 *  \brief Algorithm class for training Naive Bayes partial model in the distributed processing mode
 *  \n<a href="DAAL-REF-MULTINOMNAIVEBAYES-ALGORITHM">Multinomial naive Bayes algorithm description and usage models</a>
 *
 *  \tparam algorithmFPType  Data type to use in intermediate computations for the multinomial naive Bayes training on the first step in distributed
 *                           processing mode, double or float
 *  \tparam method           Naive Bayes training method, \ref Method
 *
 *  \par Enumerations
 *      - \ref Method Training methods for the multinomial naive Bayes on the first step in the distributed processing mode
 *
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step1Local, algorithmFPType, method> : public Online<algorithmFPType, method>
{
public:
    /**
     * Default constructor
     * \param nClasses  Number of classes
     */
    Distributed(size_t nClasses) : Online<algorithmFPType, method>::Online(nClasses) {}

    /**
     * Constructs multinomial naive Bayes training algorithm by copying input objects and parameters
     * of another multinomial naive Bayes training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step1Local, algorithmFPType, method> &other) :
        Online<algorithmFPType, method>(other)
    {}

    /**
     * Returns a pointer to the newly allocated multinomial naive Bayes training algorithm
     * with a copy of input objects and parameters of this multinomial naive Bayes training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step1Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step1Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step1Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step1Local, algorithmFPType, method>(*this);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__DISTRIBUTED_STEP2MASTER_ALGORITHMFPTTYPE_METHOD"></a>
 *  \brief Algorithm class for training naive Bayes final model on the second step in the distributed processing mode
 *  \n<a href="DAAL-REF-MULTINOMNAIVEBAYES-ALGORITHM">Multinomial naive Bayes algorithm description and usage models</a>
 *
 *  \tparam algorithmFPType  Data type to use in intermediate computations for the multinomial naive Bayes training on the second step in
 *                           distributed processing mode, double or float
 *  \tparam method           Naive Bayes training method on the second step in distributed processing mode, \ref Method
 *
 *  \par Enumerations
 *      - \ref Method Training methods for the multinomial naive Bayes algorithm
 *
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step2Master, algorithmFPType, method> : public classifier::training::Distributed
{
public:
    /**
     * Default constructor
     * \param nClasses  Number of classes
     */
    Distributed(size_t nClasses) :
        classifier::training::Distributed(),
        parameter(nClasses)
    {
        initialize();
    }

    /**
     * Constructs multinomial naive Bayes training algorithm by copying input objects and parameters
     * of another multinomial naive Bayes training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step2Master, algorithmFPType, method> &other) :
        classifier::training::Distributed(other), parameter(other.parameter.nClasses)
    {
        initialize();
        parameter = other.parameter;
    }

    virtual ~Distributed() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns the structure that contains results of Naive Bayes training
     * \return Structure that contains results of Naive Bayes training
     */
    services::SharedPtr<Result> getResult()
    {
        return services::staticPointerCast<Result, classifier::training::Result>(_result);
    }

    /**
     * Registers user-allocated memory to store results of Naive Bayes training
     * \param[in] result  Structure to store  results of Naive Bayes training
     */
    void setResult(const services::SharedPtr<Result>& result)
    {
        DAAL_CHECK(result, ErrorNullResult)
        _result = result;
        _res = _result.get();
    }

    /**
     * Resets the training results of the classification algorithm
     */
    void resetResult()
    {
        _result = services::SharedPtr<Result>(new Result());
        _res = NULL;
    }

    /**
    * Validates parameters of the finalizeCompute() method
    */
    void checkFinalizeComputeParams() DAAL_C11_OVERRIDE
    {
        services::SharedPtr<PartialResult> partialResult
            = services::staticPointerCast<PartialResult, classifier::training::PartialResult>(_partialResult);
        if(partialResult)
        {
            partialResult->check(_par, method);
            if (!_errors->isEmpty()) { return; }
        }
        else
        {
            _errors->add(services::ErrorNullResult);
            return;
        }

        services::SharedPtr<Result> result = services::staticPointerCast<Result, classifier::training::Result>(_result);
        if(result)
        {
            result->check(_pres, _par, method);
        }
        else
        {
            _errors->add(services::ErrorNullResult);
            return;
        }
    }

    /**
     * Returns a pointer to the newly allocated multinomial naive Bayes training algorithm
     * with a copy of input objects and parameters of this multinomial naive Bayes training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step2Master, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step2Master, algorithmFPType, method> >(cloneImpl());
    }

    Parameter parameter; /*!< Parameters of the distributed training algorithm */

protected:
    virtual Distributed<step2Master, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step2Master, algorithmFPType, method>(*this);
    }

    void allocateResult() DAAL_C11_OVERRIDE
    {
        services::SharedPtr<PartialResult> pres = services::staticPointerCast<PartialResult, classifier::training::PartialResult>(_partialResult);
        services::SharedPtr<Result> res = services::staticPointerCast<Result, classifier::training::Result>(_result);
        res->template allocate<algorithmFPType>(pres.get(), &parameter, (int)method);
        _res = _result.get();
    }

    void allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::SharedPtr<PartialResult> pres = services::staticPointerCast<PartialResult, classifier::training::PartialResult>(_partialResult);
        pres->template allocate<algorithmFPType>((classifier::training::InputIface *)(&input), &parameter, (int)method);
        _pres = _partialResult.get();
    }

    void initializePartialResult() DAAL_C11_OVERRIDE {}

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master, algorithmFPType, method)(&_env);
        _par = &parameter;
        _result = services::SharedPtr<Result>(new Result());
        _partialResult = services::SharedPtr<PartialResult>(new PartialResult());
    }
};
/** @} */
} // namespace interface1
using interface1::DistributedContainer;
using interface1::Distributed;

} // namespace training
} // namespace multinomial_naive_bayes
} // namespace algorithms
} //namespace daal
#endif
