/* file: MergedNumericTable.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__MERGEDNUMERICTABLE"></a>
 *  @brief Class that provides methods to access a collection of numeric tables as if they are joined by columns
 */
public class MergedNumericTable extends NumericTable {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public MergedNumericTable(DaalContext context, MergedNumericTableImpl impl) {
        super(context);
        tableImpl = impl;
    }

    /**
     * Constructs empty merged numeric table
     * @param context   Context to manage created merged numeric table
     */
    public MergedNumericTable(DaalContext context) {
        super(context);
        tableImpl = new MergedNumericTableImpl(context);
    }

    /**
     * Constructs merged numeric table from C++ merged numeric table
     *
     * @param context   Context to manage created merged numeric table
     * @param cTable    Pointer to C++ numeric table
     */
    public MergedNumericTable(DaalContext context, long cTable) {
        super(context);
        tableImpl = new MergedNumericTableImpl(context, cTable);
    }

    /**
     * Constructs merged numeric table consisting of one table
     *
     * @param context   Context to manage created merged numeric table
     * @param table     Pointer to the Numeric Table
     */
    public MergedNumericTable(DaalContext context, NumericTable table) {
        super(context);
        tableImpl = new MergedNumericTableImpl(context, table);
    }

    /**
     * Constructs merged numeric table consisting of two tables
     *
     * @param context   Context to manage created merged numeric table
     * @param first     Pointer to the first Numeric Table
     * @param second    Pointer to the second Numeric Table
     */
    public MergedNumericTable(DaalContext context, NumericTable first, NumericTable second) {
        super(context);
        tableImpl = new MergedNumericTableImpl(context, first, second);
    }

    /**
     *  Adds the table to the right of the merged numeric table
     *  \param table    Pointer to the Numeric Table
     */
    public void addNumericTable(NumericTable table) {
        ((MergedNumericTableImpl)tableImpl).addNumericTable(table);
    }
}
/** @} */
