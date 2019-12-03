/* file: pca_types_v1.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of PCA algorithm interface.
//--
*/

#ifndef __PCA_TYPES_V1_H__
#define __PCA_TYPES_V1_H__

#include "algorithms/pca/pca_types.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup pca Principal Component Analysis
 * \copydoc daal::algorithms::pca
 * @ingroup analysis
 * @{
 */
/**
 * \brief Contains classes for computing the results of the principal component analysis (PCA) algorithm
 */
namespace pca
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
* <a name="DAAL-CLASS-ALGORITHMS__PCA__BATCHPARAMETER"></a>
* \brief Class that specifies the parameters of the PCA algorithm in the batch computing mode
*/
template <typename algorithmFPType, Method method>
class BatchParameter : public BaseParameter<algorithmFPType, method>
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__BATCHPARAMETER_ALGORITHMFPTYPE_CORRELATIONDENSE"></a>
 * \brief Class that specifies the parameters of the PCA Correlation algorithm in the batch computing mode
 */
template <typename algorithmFPType>
class DAAL_EXPORT BatchParameter<algorithmFPType, correlationDense> : public BaseParameter<algorithmFPType, correlationDense>
{
public:
    /** Constructs PCA parameters */
    BatchParameter(const services::SharedPtr<covariance::BatchImpl> & covarianceForBatchParameter =
                       services::SharedPtr<covariance::Batch<algorithmFPType, covariance::defaultDense> >(
                           new covariance::Batch<algorithmFPType, covariance::defaultDense>()));

    services::SharedPtr<covariance::BatchImpl> covariance; /*!< Pointer to batch covariance */

    /**
    * Checks batch parameter of the PCA correlation algorithm
    * \return Errors detected while checking
    */
    services::Status check() const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__RESULT"></a>
 * \brief Provides methods to access results obtained with the PCA algorithm
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    Result(const Result & o);
    Result();

    virtual ~Result() {};

    /**
    * Gets the results of the PCA algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets results of the PCA algorithm
     * \param[in] id      Identifier of the result
     * \param[in] value   Pointer to the object
     */
    void set(ResultId id, const data_management::NumericTablePtr & value);

    /**
     * Allocates memory for storing partial results of the PCA algorithm
     * \param[in] input Pointer to an object containing input data
     * \param[in] parameter Algorithm parameter
     * \param[in] method Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, daal::algorithms::Parameter * parameter, const Method method);

    /**
     * Allocates memory for storing partial results of the PCA algorithm     * \param[in] partialResult Pointer to an object containing input data
     * \param[in] parameter Parameter of the algorithm
     * \param[in] method        Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::PartialResult * partialResult, daal::algorithms::Parameter * parameter,
                                          const Method method);

    /**
    * Checks the results of the PCA algorithm
    * \param[in] _input  %Input object of algorithm
    * \param[in] par     Algorithm %parameter
    * \param[in] method  Computation  method
    * \return Errors detected while checking
    */
    services::Status check(const daal::algorithms::Input * _input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks the results of the PCA algorithm
    * \param[in] pr             Partial results of the algorithm
    * \param[in] method         Computation method
    * \param[in] parameter      Algorithm %parameter
    * \return Errors detected while checking
    */
    services::Status check(const daal::algorithms::PartialResult * pr, const daal::algorithms::Parameter * parameter,
                           int method) const DAAL_C11_OVERRIDE;

protected:
    services::Status checkImpl(size_t nFeatures) const;
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

/** @} */
} // namespace interface1
} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
