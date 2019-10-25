/* file: gbt_regression_training_distributed_input.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of gradient boosted trees algorithm classes.
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_regression_training_types.h"
#include "daal_strings.h"
#include "service_math.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace training
{
namespace interface1
{

DistributedInput<step1Local>::DistributedInput() : daal::algorithms::Input(lastStep1LocalNumericTableInputId + 1) {}

/**
 * Returns an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr DistributedInput<step1Local>::get(Step1LocalNumericTableInputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Sets an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step1Local>::set(Step1LocalNumericTableInputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the input objects and parameters of gradient boosted trees model-based training
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step1Local>::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *par = static_cast<const Parameter *>(parameter);
    DAAL_CHECK(par, services::ErrorNullInput);

    const int unexpectedLayouts = (int)packed_mask;

    /* binnedData */
//    NumericTablePtr binnedDataTable = get(Step1LocalNumericTableInputId::step1BinnedData);

//    const size_t nRows = binnedDataTable->getNumberOfRows();
//    const size_t nCols = binnedDataTable->getNumberOfColumns();

//    DAAL_CHECK_STATUS_VAR(checkNumericTable(binnedDataTable.get(), gbtStep1binnedDataStr(), unexpectedLayouts, 0, 0, 0));

    /* binSizes */
//    NumericTablePtr binSizesDataTable = get(step1BinSizes);
//    DAAL_CHECK_STATUS_VAR(checkNumericTable(binSizesDataTable.get(), gbtStep1binSizesStr(), unexpectedLayouts, 0, nCols, 1));

    /* binOffsets */
//    NumericTablePtr binOffsetsDataTable = get(step1BinOffsets);
//    DAAL_CHECK_STATUS_VAR(checkNumericTable(binOffsetsDataTable.get(), gbtStep1binOffsetsStr(), unexpectedLayouts, 0, nCols, 1));

    /* treeOrder */
//    NumericTablePtr treeOrderDataTable = get(step1InputTreeOrder);
//    DAAL_CHECK_STATUS_VAR(checkNumericTable(treeOrderDataTable.get(), gbtStep1InputTreeOrderStr(), unexpectedLayouts, 0, 1, nRows));

    /* dependentVariable */
//    NumericTablePtr dependentVariableDataTable = get(step1DependentVariable);
//    DAAL_CHECK_STATUS_VAR(checkNumericTable(dependentVariableDataTable.get(), gbtStep1DependentVariableStr(), unexpectedLayouts, 0, 1, nRows));

    /* optCoeffs */
//    NumericTablePtr optCoeffsDataTable = get(step1OptCoeffs);
//    DAAL_CHECK_STATUS_VAR(checkNumericTable(optCoeffsDataTable.get(), gbtStep1optCoeffsStr(), unexpectedLayouts, 0, 2, nRows));

    /* treeStructure */
//    NumericTablePtr treeStructure = get(step1InputTreeStructure);
//    DAAL_CHECK(treeStructure, services::ErrorNullNumericTable);

    return services::Status();
}


DistributedInput<step2Local>::DistributedInput() : daal::algorithms::Input(lastStep2LocalNumericTableInputId + 1) {}

/**
 * Returns an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr DistributedInput<step2Local>::get(Step2LocalNumericTableInputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Sets an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step2Local>::set(Step2LocalNumericTableInputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the input objects and parameters of gradient boosted trees model-based training
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step2Local>::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *par = static_cast<const Parameter *>(parameter);

    return services::Status();
}


DistributedInput<step3Local>::DistributedInput() : daal::algorithms::Input(lastStep3LocalCollectionInputId + 1) {}

/**
 * Returns an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr DistributedInput<step3Local>::get(Step3LocalNumericTableInputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Returns an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step3Local>::get(Step3LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step3Local>::set(Step3LocalNumericTableInputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step3Local>::set(Step3LocalCollectionInputId id, const DataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step3Local>::add(Step3LocalCollectionInputId id, const NumericTablePtr &ptr)
{
    data_management::DataCollectionPtr collection =
            services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection) { return; }
    if (!ptr)        { return; }
    collection->push_back(ptr);
}

/**
 * Checks the input objects and parameters of gradient boosted trees model-based training
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step3Local>::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *par = static_cast<const Parameter *>(parameter);

    return services::Status();
}


DistributedInput<step4Local>::DistributedInput() : daal::algorithms::Input(lastStep4LocalCollectionInputId + 1) {}

/**
 * Returns an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr DistributedInput<step4Local>::get(Step4LocalNumericTableInputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Returns an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step4Local>::get(Step4LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step4Local>::set(Step4LocalNumericTableInputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step4Local>::set(Step4LocalCollectionInputId id, const DataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the input objects and parameters of gradient boosted trees model-based training
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step4Local>::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *par = static_cast<const Parameter *>(parameter);

    return services::Status();
}


DistributedInput<step5Local>::DistributedInput() : daal::algorithms::Input(lastStep5LocalCollectionInputId + 1) { }

/**
 * Returns an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr DistributedInput<step5Local>::get(Step5LocalNumericTableInputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Returns an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step5Local>::get(Step5LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step5Local>::set(Step5LocalNumericTableInputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step5Local>::set(Step5LocalCollectionInputId id, const DataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step5Local>::add(Step5LocalCollectionInputId id, const NumericTablePtr &ptr)
{
    data_management::DataCollectionPtr collection =
            services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection) { return; }
    if (!ptr)        { return; }
    collection->push_back(ptr);
}

/**
 * Checks the input objects and parameters of gradient boosted trees model-based training
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step5Local>::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *par = static_cast<const Parameter *>(parameter);

    return services::Status();
}


DistributedInput<step6Local>::DistributedInput() : daal::algorithms::Input(lastStep6LocalCollectionInputId + 1) {}

/**
 * Returns an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr DistributedInput<step6Local>::get(Step6LocalNumericTableInputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Returns an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step6Local>::get(Step6LocalCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step6Local>::set(Step6LocalNumericTableInputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step6Local>::set(Step6LocalCollectionInputId id, const DataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step6Local>::add(Step6LocalCollectionInputId id, const NumericTablePtr &ptr)
{
    data_management::DataCollectionPtr collection =
            services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection) { return; }
    if (!ptr)        { return; }
    collection->push_back(ptr);
}

/**
 * Checks the input objects and parameters of gradient boosted trees model-based training
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step6Local>::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *par = static_cast<const Parameter *>(parameter);

    return services::Status();
}

} // namespace interface1
} // namespace training
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
