/* file: multinomial_naive_bayes_training_online.h */
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
//  Implementation of the interface for multinomial naive Bayes model-based training
//  in the online processing mode
//--
*/

#ifndef __NAIVE_BAYES_TRAINING_ONLINE_H__
#define __NAIVE_BAYES_TRAINING_ONLINE_H__

#include "algorithms/algorithm.h"
#include "multinomial_naive_bayes_training_types.h"
#include "algorithms/classifier/classifier_training_online.h"

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
 * @defgroup multinomial_naive_bayes_training_online Online
 * @ingroup multinomial_naive_bayes_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__ONLINECONTAINER"></a>
 *  \brief Class containing computation methods for naive Bayes training in the online processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the naive Bayes in the online processing mode, double or float
 * \tparam method           Naive Bayes computation method, \ref Method
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT OnlineContainer : public TrainingContainerIface<online>
{
public:
    /**
     * Constructs a container for multinomial naive Bayes model-based training with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~OnlineContainer();

    /**
     * Computes a partial result of naive Bayes model-based training
     * in the online processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of naive Bayes model-based training
     * in the online processing mode
     *
     * \return Status of computations
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__ONLINE"></a>
 *  \brief Algorithm class for training naive Bayes model
 *  <!-- \n<a href="DAAL-REF-MULTINOMNAIVEBAYES-ALGORITHM">Multinomial naive Bayes algorithm description and usage models</a> -->
 *
 *  \tparam algorithmFPType  Data type to use in intermediate computations for multinomial naive Bayes training in the online processing mode,
 *                           double or float
 *  \tparam method           Computation method, \ref Method
 *
 *  \par Enumerations
 *      - \ref Method %Training methods for the multinomial naive Bayes algorithm
 *
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Online : public classifier::training::Online
{
public:
    typedef classifier::training::Online super;

    typedef typename super::InputType                                    InputType;
    typedef algorithms::multinomial_naive_bayes::Parameter               ParameterType;
    typedef algorithms::multinomial_naive_bayes::training::Result        ResultType;
    typedef algorithms::multinomial_naive_bayes::training::PartialResult PartialResultType;

    /**
     * Default constructor
     * \param nClasses  Number of classes
     */
    Online(size_t nClasses) : parameter(nClasses)
    {
        initialize();
    }

    /**
     * Constructs multinomial naive Bayes training algorithm by copying input objects and parameters
     * of another multinomial naive Bayes training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Online(const Online<algorithmFPType, method> &other) :
        classifier::training::Online(other), parameter(other.parameter)
    {
        initialize();
    }

    virtual ~Online() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns the structure that contains results of Naive Bayes training
     * \return Structure that contains results of Naive Bayes training
     */
    ResultPtr getResult()
    {
        return services::staticPointerCast<ResultType, classifier::training::Result>(_result);
    }

    /**
     * Registers user-allocated memory to store results of Naive Bayes training
     * \param[in] result  Structure to store  results of Naive Bayes training
     *
     * \return Status of computations
     */
    services::Status setResult(const ResultPtr& result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Resets the training results of the classification algorithm
     */
    void resetResult()
    {
        _result.reset(new ResultType());
        _res = NULL;
    }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    PartialResultPtr getPartialResult() { return PartialResultType::cast(_partialResult); }

    /**
     * Returns a pointer to the newly allocated multinomial naive Bayes training algorithm
     * with a copy of input objects and parameters of this multinomial naive Bayes training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Online<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Online<algorithmFPType, method> >(cloneImpl());
    }

    ParameterType parameter;                /*!< \ref interface1::Parameter "Parameters" of the training */

protected:

    virtual Online<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Online<algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        PartialResultPtr pres = getPartialResult();
        ResultPtr res = services::staticPointerCast<ResultType, classifier::training::Result>(_result);
        services::Status s = res->template allocate<algorithmFPType>(pres.get(), &parameter, (int)method);
        _res = _result.get();
        return s;
    }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        PartialResultPtr pres = getPartialResult();
        services::Status s = pres->template allocate<algorithmFPType>((classifier::training::InputIface *)(&input), &parameter, (int)method);
        _pres = _partialResult.get();
        return s;
    }

    services::Status initializePartialResult() DAAL_C11_OVERRIDE
    {
        PartialResultPtr pres = getPartialResult();
        services::Status s = pres->template initialize<algorithmFPType>((classifier::training::InputIface *)(&input), &parameter, (int)method);
        _pres = _partialResult.get();
        return s;
    }

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(online, OnlineContainer, algorithmFPType, method)(&_env);
        _par = &parameter;
        _result.reset(new ResultType());
        _partialResult.reset(new PartialResultType());
    }
};
/** @} */
} // namespace interface1
using interface1::OnlineContainer;
using interface1::Online;

} // namespace training
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
#endif
