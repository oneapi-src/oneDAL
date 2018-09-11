/* file: pca_batch.h */
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
//  Implementation of the interface for the PCA algorithm in the batch processing mode
//--
*/

#ifndef __PCA_BATCH_H__
#define __PCA_BATCH_H__

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

namespace interface2
{
/**
 * @defgroup pca_batch Batch
 * @ingroup pca
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__BATCHCONTAINER"></a>
 * \brief Class containing methods to compute the results of the PCA algorithm */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public AnalysisContainerIface<batch>
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__BATCHCONTAINER_ALGORITHMFPTYPE_CORRELATIONDENSE_CPU"></a>
 * \brief Class containing methods to compute the results of the PCA algorithm */
template<typename algorithmFPType, CpuType cpu>
class DAAL_EXPORT BatchContainer<algorithmFPType, correlationDense, cpu> : public AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the PCA algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the PCA algorithm in the batch processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__BATCHCONTAINER_ALGORITHMFPTYPE_SVDDENSE_CPU"></a>
 * \brief Class containing methods to compute the results of the PCA algorithm */
template<typename algorithmFPType, CpuType cpu>
class DAAL_EXPORT BatchContainer<algorithmFPType, svdDense, cpu> : public AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the PCA algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the PCA algorithm in the batch processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__BATCH"></a>
 * \brief Computes the results of the PCA algorithm
 * <!-- \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for PCA, double or float
 * \tparam method           PCA computation method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods for the algorithm
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = correlationDense>
class DAAL_EXPORT Batch : public Analysis<batch>
{
public:
    typedef algorithms::pca::Input                                   InputType;
    typedef algorithms::pca::BatchParameter<algorithmFPType, method> ParameterType;
    typedef algorithms::pca::Result                                  ResultType;

    /** Default constructor */
    Batch()
    {
        initialize();
    }

    /**
     * Constructs a PCA algorithm by copying input objects and parameters of another PCA algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : input(other.input), parameter(other.parameter)
    {
        initialize();
    }

    ~Batch() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return method; };

    /**
     * Registers user-allocated memory to store the results of the PCA algorithm
     * \param[in] res    Structure for storing the results of the PCA algorithm
     */
    services::Status setResult(const ResultPtr& res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains final results of the PCA algorithm
     * \return Structure that contains final results of the PCA
     */
    ResultPtr getResult() { return _result; }

    /**
     * Returns a pointer to the newly allocated PCA algorithm
     * with a copy of input objects and parameters of this PCA algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    InputType input; /*!< Input data structure */
    BatchParameter<algorithmFPType, method> parameter; /*!< Parameters */

protected:
    ResultPtr _result;

    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, &parameter, method);
        _res = _result.get();
        return s;
    }

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }
};
/** @} */
} // namespace interface1
using interface2::BatchContainer;
using interface2::Batch;

}
}
}
#endif
