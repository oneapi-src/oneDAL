/* file: em_gmm_init_types.h */
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
//  Implementation of the EM for GMM initialization interface.
//--
*/

#ifndef __EM_GMM_INIT_TYPES_H__
#define __EM_GMM_INIT_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
/**
 * @defgroup em_gmm Expectation-Maximization
 * \copydoc daal::algorithms::em_gmm
 * @ingroup analysis
 * @defgroup em_gmm_init Initialization
 * \copydoc daal::algorithms::em_gmm::init
 * @ingroup em_gmm
 * @{
 */
/**
 * \brief Contains classes for the EM for GMM initialization algorithm
 */
namespace init
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__EM_GMM__INIT__INPUTID"></a>
 * Available identifiers of input objects for the computation of initial values for the EM for GMM algorithm
 */
enum InputId
{
    data = 0               /*!< %Input data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__EM_GMM__INIT__RESULTID"></a>
 * Available identifiers of results for the computation of initial values for the EM for GMM algorithm
 */
enum ResultId
{
    weights = 0,          /*!< Weights */
    means   = 1           /*!< Means */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__EM_GMM__INIT__RESULTCOVARIANCESID"></a>
 * Available identifiers of initialized covariances for the EM for GMM algorithm
 */
enum ResultCovariancesId
{
    covariances = 2       /*!< %Collection of initialized covariances */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__EM_GMM__INIT__METHOD"></a>
 * Available methods for the computation of initial values for the EM for GMM algorithm
 */
enum Method
{
    defaultDense = 0       /*!< Default: performance-oriented method. */
};


/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__EM_GMM__INIT__PARAMETER"></a>
 * \brief %Parameter for the computation of initial values for the EM for GMM algorithm
 *
 * \snippet em/em_gmm_init_types.h Parameter source code
 */
/* [Parameter source code] */

struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(size_t nComponents, size_t nTrials = 20, size_t nIterations = 10, size_t seed = 777,
              double accuracyThreshold = 1.0e-04);

    virtual ~Parameter() {}

    /**
     * Checks the correctness of the parameter
     */
    virtual void check() const;

    size_t nComponents;       /*!< Number of components in the Gaussian mixture model */
    size_t nTrials;           /*!< Number of trials of short EM runs */
    size_t nIterations;       /*!< Number of iterations in every short EM run */
    size_t seed;              /*!< Seed for randomly generating data points to start the initialization of short EM */
    double accuracyThreshold; /*!< Threshold for the termination of the algorithm */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__INIT__INPUT"></a>
 * \brief %Input objects for the computation of initial values for the EM for GMM algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input();

    virtual ~Input() {}

    /**
    * Sets the input for the EM for GMM algorithm
    * \param[in] id    Identifier of the input
    * \param[in] ptr   Pointer to the value
    */
    void set(InputId id, const data_management::NumericTablePtr &ptr);

    /**
    * Returns the input NumericTable for the computation of initial values for the EM for GMM algorithm
    * \param[in] id    Identifier of the input NumericTable
    * \return          %Input NumericTable that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(InputId id) const;

    /**
    * Checks input for the computation of initial values for the EM for GMM algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Method of the algorithm
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__INIT__RESULT"></a>
 * \brief %Results obtained with the compute() method of the initialization of the EM for GMM algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    /** Default constructor */
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory for storing initial values for results of the EM for GMM algorithm
     * \param[in] input        Pointer to the input structure
     * \param[in] parameter    Pointer to the parameter structure
     * \param[in] method       Method of the algorithm
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Sets the result for the computation of initial values for the EM for GMM algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the numeric table for the result
     */
    void set(ResultId id, const data_management::NumericTablePtr &ptr);

    /**
     * Sets the covariance collection for initialization of EM for GMM algorithm
     * \param[in] id    Identifier of the collection of covariances
     * \param[in] ptr   Pointer to the collection of covariances
     */
    void set(ResultCovariancesId id, const data_management::DataCollectionPtr &ptr);

    /**
     * Returns the result for the computation of initial values for the EM for GMM algorithm
     * \param[in] id   %Result identifier
     * \return         %Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Returns the collection of initialized covariances for the EM for GMM algorithm
     * \param[in] id   Identifier of the collection of covariances
     * \return         Collection of covariances that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(ResultCovariancesId id) const;

    /**
     * Returns the covariance with a given index from the collection of initialized covariances
     * \param[in] id    Identifier of the collection of covariances
     * \param[in] index Index of the covariance to be returned
     * \return          Pointer to the table with initialized covariances
     */
    data_management::NumericTablePtr get(ResultCovariancesId id, size_t index) const;

    /**
    * Checks the result of the computation of initial values for the EM for GMM algorithm
    * \param[in] input   %Input of the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Method of the algorithm
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_EM_GMM_INIT_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for a deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
/** @} */
/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::Result;

} // namespace init
} // namespace em_gmm
} // namespace algorithm
} // namespace daal
#endif
