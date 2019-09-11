/* file: MergedNumericTable.java */
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

/**
 * @ingroup numeric_tables
 * @{
 */
package com.intel.daal.data_management.data;

import com.intel.daal.utils.*;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Vector;

import com.intel.daal.data_management.data.DataCollection;
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
