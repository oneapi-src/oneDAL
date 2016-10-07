/* file: apriori_types.h */
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
//  Association rules parameter structure
//--
*/

#ifndef __APRIORI_TYPES_H__
#define __APRIORI_TYPES_H__

#include "services/daal_defines.h"
#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup association_rules Association Rules
 * \copydoc daal::algorithms::association_rules
 * @ingroup analysis
 * @{
 */
/**
 * \brief Contains classes for the association rules algorithm
 */
namespace association_rules
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ASSOCIATION_RULES__METHOD"></a>
 * Available methods for finding large itemsets and association rules
 */
enum Method
{
    apriori = 0,         /*!< Apriori method */
    defaultDense = 0     /*!< Apriori default method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__ASSOCIATION_RULES__ITEMSETSORDER"></a>
 * Available sort order options for resulting itemsets
 */
enum ItemsetsOrder
{
    itemsetsUnsorted = 0,           /*!< Unsorted */
    itemsetsSortedBySupport = 1     /*!< Sorted by the support value */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__ASSOCIATION_RULES__RULESORDER"></a>
 * Available sort order options for resulting association rules
 */
enum RulesOrder
{
    rulesUnsorted = 0,              /*!< Unsorted */
    rulesSortedByConfidence = 1     /*!< Sorted by the confidence value */
};


/**
 * <a name="DAAL-ENUM-ALGORITHMS__ASSOCIATION_RULES__INPUTID"></a>
 * Available identifiers of input objects for the association rules algorithm
 */
enum InputId
{
    data = 0           /*!< %Input data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__ASSOCIATION_RULES__RESULTID"></a>
 * Available identifiers of results for the association rules algorithm
 */
enum ResultId
{
    largeItemsets        = 0,       /*!< Large itemsets            */
    largeItemsetsSupport = 1,       /*!< Support of large itemsets */
    antecedentItemsets   = 2,       /*!< Antecedent itemsets       */
    consequentItemsets   = 3,       /*!< Consequent itemsets       */
    confidence           = 4        /*!< Confidence                */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__ASSOCIATION_RULES__PARAMETER"></a>
 * \brief Parameters for the association rules compute() method
 *
 * \snippet association_rules/apriori_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(double minSupport = 0.01, double minConfidence = 0.6, size_t nUniqueItems = 0, size_t nTransactions = 0,
              bool discoverRules = true, ItemsetsOrder itemsetsOrder = itemsetsUnsorted,
              RulesOrder rulesOrder = rulesUnsorted, size_t minSize = 0, size_t maxSize = 0);

    double         minSupport;          /*!< Minimum support    0.0 <= minSupport    < 1.0 */
    double         minConfidence;       /*!< Minimum confidence 0.0 <= minConfidence < 1.0 */
    size_t         nUniqueItems;        /*!< Number of unique items */
    size_t         nTransactions;       /*!< Number of transactions */
    bool           discoverRules;       /*!< Flag. If true, association rules are built from large itemsets */
    ItemsetsOrder  itemsetsOrder;       /*!< Format of the resulting itemsets */
    RulesOrder     rulesOrder;          /*!< Format of the resulting association rules */
    size_t         minItemsetSize;      /*!< Minimum number of items in a large itemset */
    size_t         maxItemsetSize;      /*!< Maximum number of items in a large itemset.
                                             Set to zero to not limit the upper boundary for the size of large itemsets */

    /**
     * Checks parameters of the association rules algorithm
     */
    void check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__INPUT"></a>
 * \brief %Input for the association rules algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input();

    virtual ~Input() {}

    /**
     * Returns the input object of the association rules algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets the input object of the association rules algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks parameters of the association rules algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method of the algorithm
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__RESULT"></a>
 * \brief Results obtained with the compute() method of the association rules algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory for storing Association Rules algorithm results
     * \param[in] input         Pointer to input structure
     * \param[in] parameter     Pointer to parameter structure
     * \param[in] method        Computation method of the algorithm
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Returns the final result of the association rules algorithm
     * \param[in] id   Identifier of the result
     * \return         Final result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the final result of the association rules algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks the result of the association rules algorithm
     * \param[in] input   %Input of algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    int getSerializationTag() DAAL_C11_OVERRIDE { return SERIALIZATION_ASSOCIATION_RULES_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for the serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for the deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    void checkNT(data_management::NumericTablePtr nt, services::SharedPtr<services::ErrorCollection> errors, const char *description, size_t nColumns) const;

    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::Result;

} // namespace association_rules
} // namespace algorithm
} // namespace daal
#endif
