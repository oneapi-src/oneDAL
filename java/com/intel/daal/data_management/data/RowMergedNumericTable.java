/* file: RowMergedNumericTable.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

/**
 * @ingroup numeric_tables
 * @{
 */
package com.intel.daal.data_management.data;

import com.intel.daal.utils.*;

import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__ROWMERGEDNUMERICTABLE"></a>
 *  @brief Class that provides methods to access a collection of numeric tables as if they are joined by rows
 */
public class RowMergedNumericTable extends NumericTable {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public RowMergedNumericTable(DaalContext context, RowMergedNumericTableImpl impl) {
        super(context);
        tableImpl = impl;
    }

    /**
     * Constructs empty row merged numeric table
     *
     * @param context   Context to manage created row merged numeric table
     */
    public RowMergedNumericTable(DaalContext context) {
        super(context);
        tableImpl = new RowMergedNumericTableImpl(context);
    }

    /**
     * Constructs row merged numeric table from C++ row merged numeric table
     *
     * @param context   Context to manage created row merged numeric table
     * @param cTable    Pointer to C++ numeric table
     */
    public RowMergedNumericTable(DaalContext context, long cTable) {
        super(context);
        tableImpl = new RowMergedNumericTableImpl(context, cTable);
    }

    /**
     * Constructs row merged numeric table consisting of one table
     *
     * @param context   Context to manage created row merged numeric table
     * @param table     Pointer to the Numeric Table
     */
    public RowMergedNumericTable(DaalContext context, NumericTable table) {
        super(context);
        tableImpl = new RowMergedNumericTableImpl(context, table);
    }

    /**
     *  Adds the table to the right of the rowmerged numeric table
     *  \param table    Pointer to the Numeric Table
     */
    public void addNumericTable(NumericTable table) {
        ((RowMergedNumericTableImpl)tableImpl).addNumericTable(table);
    }
}
/** @} */
