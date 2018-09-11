/* file: AOSNumericTable.java */
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

/**
 * @ingroup numeric_tables
 * @{
 */
package com.intel.daal.data_management.data;

import com.intel.daal.utils.*;
import java.lang.reflect.Field;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__AOSNUMERICTABLE"></a>
 * @brief Class that provides methods to access data that is stored as a contiguous array
 *         of heterogeneous feature vectors,  and each feature vector is represented
 *         with a data structure.
 *         Therefore, the data is represented as an Array Of Structures(AOS).
 */
public class AOSNumericTable extends NumericTable {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public AOSNumericTable(DaalContext context, AOSNumericTableImpl impl) {
        super(context);
        tableImpl = impl;
    }

    /**
     * Constructor for empty Numeric Table with predefined class for the feature vectors and given number of feature vectors
     *
     * @param context   Context to manage created AOS numeric table
     * @param cls       Class containing expected array elements
     * @param nVectors  The number of rows in the table
     */
    public AOSNumericTable(DaalContext context, Class<?> cls, long nVectors) {
        super(context);
        tableImpl = new AOSNumericTableImpl(context, cls, nVectors);
    }

    /**
     * Constructs Numeric Table from the array of objects representing feature vectors
     *
     * @param context Context to manage created AOS numeric table
     * @param ptr     Array of objects to associate with the Numeric Table
     */
    public AOSNumericTable(DaalContext context, Object[] ptr) {
        super(context);
        tableImpl = new AOSNumericTableImpl(context, ptr);
    }

    /**
     * Array of objects associated with the table
     */
    public void setArray(Object[] arr) {
        ((AOSNumericTableImpl)tableImpl).setArray(arr);
    }

    /**
     * Returns the array of objects associated with the table
     *
     * @return Array of objects
     */
    public Object[] getArray() {
        return ((AOSNumericTableImpl)tableImpl).getArray();
    }
}
/** @} */
