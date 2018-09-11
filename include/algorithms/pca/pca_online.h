/* file: pca_online.h */
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
//  Implementation of the interface for the PCA algorithm in the online processing mode
//--
*/

#ifndef __PCA_ONLINE_H__
#define __PCA_ONLINE_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "services/daal_memory.h"
#include "algorithms/pca/pca_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{

namespace interface1
{
/**
 * @defgroup pca_online Online
 * @ingroup pca
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINECONTAINER"></a>
 * \brief Class containing methods to compute the result of the PCA algorithm
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT OnlineContainer : public AnalysisContainerIface<online> {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINECONTAINER_ALGORITHMFPTYPE_CORRELATIONDENSE_CPU"></a>
 * \brief Class containing methods to compute the result of the PCA algorithm
 */
template<typename algorithmFPType, CpuType cpu>
class DAAL_EXPORT OnlineContainer<algorithmFPType, correlationDense, cpu> : public AnalysisContainerIface<online>
{
public:
    /**
     * Constructs a container for the PCA algorithm with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~OnlineContainer();

    /**
     * Computes a partial result of the PCA algorithm in the online processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the PCA algorithm in the online processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINECONTAINER_ALGORITHMFPTYPE_SVDDENSE_CPU"></a>
 * \brief Class containing methods to compute the results of the PCA algorithm
 */
template<typename algorithmFPType, CpuType cpu>
class DAAL_EXPORT OnlineContainer<algorithmFPType, svdDense, cpu> : public AnalysisContainerIface<online>
{
public:
    /**
     * Constructs a container for the PCA algorithm with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~OnlineContainer();

    /**
     * Computes a partial result of the PCA algorithm in the online processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the PCA algorithm in the online processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINE"></a>
 * \brief Computes the results of the PCA algorithm
 * <!-- \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the PCA algorithm, double or float
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = correlationDense>
class DAAL_EXPORT Online : public Analysis<online> {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINE_ALGORITHMFPTYPE_CORRELATIONDENSE"></a>
 * \brief Computes the results of the PCA Correlation algorithm
 * <!-- \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the PCA algorithm, double or float
 */
template<typename algorithmFPType>
class DAAL_EXPORT Online<algorithmFPType, correlationDense> : public Analysis<online>
{
public:
    typedef algorithms::pca::Input                                              InputType;
    typedef algorithms::pca::OnlineParameter<algorithmFPType, correlationDense> ParameterType;
    typedef algorithms::pca::Result                                             ResultType;
    typedef algorithms::pca::PartialResult<correlationDense>                    PartialResultType;

    /** Default constructor */
    Online()
    {
        initialize();
    }

    /**
     * Constructs a PCA algorithm by copying input objects and parameters of another PCA algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Online(const Online<algorithmFPType, correlationDense> &other) : input(other.input), parameter(other.parameter)
    {
        initialize();
    }

    ~Online() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    int getMethod() const DAAL_C11_OVERRIDE { return(int)correlationDense; }

    /**
     * Registers user-allocated  memory to store the results of the PCA algorithm
     * \param[in] partialResult    Structure for storing partial result of the PCA algorithm
     */
    services::Status setPartialResult(const services::SharedPtr<PartialResult<correlationDense> >& partialResult)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult);
        _partialResult = partialResult;
        _pres = _partialResult.get();
        return services::Status();
    }

    /**
     * Registers user-allocated memory to store the results of the PCA algorithm
     * \param[in] res    Structure to store the results of the PCA algorithm
     */
    services::Status setResult(const ResultPtr& res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains partial results of the PCA algorithm
     * \return Structure that contains partial results of the PCA algorithm
     */
    services::SharedPtr<PartialResult<correlationDense> > getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Returns structure that contains the results of the PCA algorithm
     * \return Structure that contains the results of the PCA algorithm
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Returns a pointer to the newly allocated PCA algorithm
     * with a copy of input objects and parameters of this PCA algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Online<algorithmFPType, correlationDense> > clone() const
    {
        return services::SharedPtr<Online<algorithmFPType, correlationDense> >(cloneImpl());
    }

    InputType input; /*!< Input data structure */
    OnlineParameter<algorithmFPType, correlationDense> parameter; /*!< Parameters */

protected:
    services::SharedPtr<PartialResult<correlationDense> > _partialResult;
    ResultPtr _result;

    virtual Online<algorithmFPType, correlationDense> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Online<algorithmFPType, correlationDense>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(_pres, &parameter, correlationDense);
        _res = _result.get();
        return s;
    }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, &parameter, correlationDense);
        _pres = _partialResult.get();
        return s;
    }

    services::Status initializePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->initialize<algorithmFPType>(&input, &parameter, correlationDense);
        _pres = _partialResult.get();
        return s;
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(online, OnlineContainer, algorithmFPType, correlationDense)(&_env);
        _in = &input;
        _par = &parameter;
        _partialResult.reset(new PartialResult<correlationDense>());
        _result.reset(new ResultType());
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINE_ALGORITHMFPTYPE_SVDDENSE"></a>
 * \brief Computes the results of the PCA SVD algorithm
 * <!-- \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the PCA algorithm, double or float
 */
template<typename algorithmFPType>
class DAAL_EXPORT Online<algorithmFPType, svdDense> : public Analysis<online>
{
public:
    typedef algorithms::pca::Input                                      InputType;
    typedef algorithms::pca::OnlineParameter<algorithmFPType, svdDense> ParameterType;
    typedef algorithms::pca::Result                                     ResultType;
    typedef algorithms::pca::PartialResult<svdDense>                    PartialResultType;

    /** Default constructor */
    Online()
    {
        initialize();
    }

    /**
     * Constructs a PCA algorithm by copying input objects and parameters of another PCA algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Online(const Online<algorithmFPType, svdDense> &other) : input(other.input), parameter(other.parameter)
    {
        initialize();
    }

    ~Online() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    int getMethod() const DAAL_C11_OVERRIDE { return(int)svdDense; }

    /**
     * Registers user-allocated  memory to store the results of the PCA algorithm
     * \param[in] partialResult    Structure for storing partial result of the PCA algorithm
     */
    services::Status setPartialResult(const services::SharedPtr<PartialResult<svdDense> >& partialResult)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult);
        _partialResult = partialResult;
        _pres = _partialResult.get();
        return services::Status();
    }

    /**
     * Registers user-allocated memory to store the results of the PCA algorithm
     * \param[in] res    Structure to store the results of the PCA algorithm
     */
    services::Status setResult(const ResultPtr& res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains partial results of the PCA algorithm
     * \return Structure that contains partial results of the PCA algorithm
     */
    services::SharedPtr<PartialResult<svdDense> > getPartialResult()
    {
        return _partialResult;
    }

    /**
     * Returns structure that contains the results of the PCA algorithm
     * \return Structure that contains the results of the PCA algorithm
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Returns a pointer to the newly allocated PCA algorithm
     * with a copy of input objects and parameters of this PCA algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Online<algorithmFPType, svdDense> > clone() const
    {
        return services::SharedPtr<Online<algorithmFPType, svdDense> >(cloneImpl());
    }

    InputType input; /*!< Input data structure */
    OnlineParameter<algorithmFPType, svdDense> parameter; /*!< Parameters */

protected:
    services::SharedPtr<PartialResult<svdDense> > _partialResult;
    ResultPtr _result;

    virtual Online<algorithmFPType, svdDense> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Online<algorithmFPType, svdDense>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(_pres, &parameter, svdDense);
        _res = _result.get();
        return s;
    }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, &parameter, svdDense);
        _pres = _partialResult.get();
        return s;
    }

    services::Status initializePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->initialize<algorithmFPType>(&input, &parameter, svdDense);
        _pres = _partialResult.get();
        return s;
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(online, OnlineContainer, algorithmFPType, svdDense)(&_env);
        _in = &input;
        _par = &parameter;
        _partialResult.reset(new PartialResult<svdDense>());
        _result.reset(new ResultType());
    }
};
/** @} */
} // namespace interface1
using interface1::OnlineContainer;
using interface1::Online;

}
}
}
#endif
