/* file: zscore_types_v1.h */
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
//  Definition of common types of z-score normalization.
//--
*/

#ifndef __ZSCORE_TYPES_V1_H__
#define __ZSCORE_TYPES_V1_H__

#include "zscore_types.h"
#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/moments/low_order_moments_batch.h"
#include "services/daal_defines.h"
#include "algorithms/algorithm.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup normalization Normalization
 * \copydoc daal::algorithms::normalization
 * @ingroup analysis
 * @{
 */
/**
 * \brief Contains classes to run the z-score normalization algorithms
 */
namespace normalization
{
/**
 * @defgroup zscore Z-score
 * \copydoc daal::algorithms::normalization::zscore
 * @ingroup normalization
 * @{
 */
/**
* \brief Contains classes for computing the z-score normalization
*/
namespace zscore
{
/**
* \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{
/**
* <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__PARAMETER"></a>
* \brief Class that specifies the parameters of the algorithm in the batch computing mode
*/
template <typename algorithmFPType, Method method>
class DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{};

// /**
//  * <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__PARAMETER"></a>
//  * \brief Class that specifies the parameters of the default algorithm in the batch computing mode
//  */
template <typename algorithmFPType>
class DAAL_EXPORT Parameter<algorithmFPType, defaultDense> : public daal::algorithms::Parameter
{
public:
    /** Constructs z-score normalization parameters */
    Parameter(const services::SharedPtr<low_order_moments::BatchImpl> & momentsForParameter =
                  services::SharedPtr<low_order_moments::Batch<algorithmFPType, low_order_moments::defaultDense> >(
                      new low_order_moments::Batch<algorithmFPType, low_order_moments::defaultDense>()));

    services::SharedPtr<low_order_moments::BatchImpl> moments; /*!< Pointer to the algorithm that computes the low order moments */

    /**
                                                               * Check the correctness of the %Parameter object
                                                               *
                                                               * \return Status of computations
                                                               */
    virtual services::Status check() const DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__RESULT"></a>
* \brief Provides methods to access final results obtained with the compute() method of the
*        z-score normalization algorithm in the batch processing mode
*/
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    Result(const Result & o);
    Result();

    virtual ~Result() {};

    /**
    * Allocates memory to store final results of the z-score normalization algorithms
    * \param[in] input     Input objects for the z-score normalization algorithm
    * \param[in] method    Algorithm computation method
    *
    * \return Status of computations
    */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const int method);

    /**
    * Returns the final result of the z-score normalization algorithm
    * \param[in] id   Identifier of the final result, daal::algorithms::normalization::zscore::ResultId
    * \return         Final result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
    * Sets the Result object of the z-score normalization algorithm
    * \param[in] id        Identifier of the Result object
    * \param[in] value     Pointer to the Result object
    */
    void set(ResultId id, const data_management::NumericTablePtr & value);

    /**
    * Checks the correctness of the Result object
    * \param[in] in     Pointer to the input object
    * \param[in] par    Pointer to the parameter object
    * \param[in] method Algorithm computation method
    *
    * \return Status of computations
    */
    services::Status check(const daal::algorithms::Input * in, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

/** @} */
/** @} */
} // namespace interface1

} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
#endif // __ZSCORE_TYPES_V1_H__
