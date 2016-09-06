/** file numeric_table.cpp */
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

#include "numeric_table.h"
#include "homogen_numeric_table.h"

/**
 * Checks the correctness of this numeric table
 * \param[in]  nt                The numeric table to check
 * \param[out] errors            The collection of errors
 * \param[in]  description       Additional information about error
 * \param[in]  unexpectedLayouts The bit mask of invalid layouts for this numeric table.
 * \param[in]  expectedLayouts   The bit mask of valid layouts for this numeric table.
 * \param[in]  nColumns          Required number of columns.
 *                               nColumns = 0 means that required number of columns is not specified.
 * \param[in]  nRows             Required number of rows.
 *                               nRows = 0 means that required number of rows is not specified.
 * \return                       Check status: True if the table satisfies the requirements, false otherwise.
 */
bool daal::data_management::checkNumericTable(const NumericTable *nt, services::ErrorCollection *errors, const char *description,
                                              const int unexpectedLayouts, const int expectedLayouts, size_t nColsValid, size_t nRowsValid)
{
    using namespace daal::services;

    if (nt == 0)
    {
        SharedPtr<Error> error = SharedPtr<Error>(new Error(ErrorNullNumericTable));
        error->addStringDetail(ArgumentName, description);
        errors->add(error);
        return false;
    }

    size_t nColumns = nt->getNumberOfColumns();
    size_t nRows = nt->getNumberOfRows();

    if(unexpectedLayouts != 0)
    {
        int state = (int)nt->getDataLayout() & unexpectedLayouts;

        if(state != 0)
        {
            SharedPtr<Error> error = SharedPtr<Error>(new Error(ErrorIncorrectTypeOfNumericTable));
            error->addStringDetail(ArgumentName, description);
            errors->add(error);
            return false;
        }
    }

    if(expectedLayouts != 0)
    {
        int state = (int)nt->getDataLayout() & expectedLayouts;

        if(state == 0)
        {
            SharedPtr<Error> error = SharedPtr<Error>(new Error(ErrorIncorrectTypeOfNumericTable));
            error->addStringDetail(ArgumentName, description);
            errors->add(error);
            return false;
        }
    }

    if (nColsValid != 0 && nColumns != nColsValid)
    {
        SharedPtr<Error> error = SharedPtr<Error>(new Error(ErrorIncorrectNumberOfColumns));
        error->addStringDetail(ArgumentName, description);
        errors->add(error);
        return false;
    }
    if (nRowsValid != 0 && nRows != nRowsValid)
    {
        SharedPtr<Error> error = SharedPtr<Error>(new Error(ErrorIncorrectNumberOfRows));
        error->addStringDetail(ArgumentName, description);
        errors->add(error);
        return false;
    }

    return nt->check(errors, description);
}

void daal::data_management::NumericTable::allocateBasicStatistics()
{
    size_t ncols = getNumberOfColumns();
    if(basicStatistics.get(NumericTable::minimum).get() == NULL || basicStatistics.get(NumericTable::minimum)->getNumberOfColumns() != ncols)
    {
        basicStatistics.set(NumericTable::minimum,
            NumericTablePtr(new HomogenNumericTable<double>(ncols, 1, NumericTableIface::doAllocate)));
    }
    if(basicStatistics.get(NumericTable::maximum).get() == NULL || basicStatistics.get(NumericTable::maximum)->getNumberOfColumns() != ncols)
    {
        basicStatistics.set(NumericTable::maximum,
            NumericTablePtr(new HomogenNumericTable<double>(ncols, 1, NumericTableIface::doAllocate)));
    }
    if(basicStatistics.get(NumericTable::sum).get() == NULL || basicStatistics.get(NumericTable::sum)->getNumberOfColumns() != ncols)
    {
        basicStatistics.set(NumericTable::sum,
            NumericTablePtr(new HomogenNumericTable<double>(ncols, 1, NumericTableIface::doAllocate)));
    }
    if(basicStatistics.get(NumericTable::sumSquares).get() == NULL ||
        basicStatistics.get(NumericTable::sumSquares)->getNumberOfColumns() != ncols)
    {
        basicStatistics.set(NumericTable::sumSquares,
            NumericTablePtr(new HomogenNumericTable<double>(ncols, 1, NumericTableIface::doAllocate)));
    }
}
