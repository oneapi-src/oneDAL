/* file: PackedTriangularMatrixImpl.java */
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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.PackedTriangularMatrix;

abstract class PackedTriangularMatrixImpl extends NumericTableImpl {
    protected Class<? extends Number> type;

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the packed triangular matrix
     * @param context   Context to manage the packed triangular matrix
     */
    public PackedTriangularMatrixImpl(DaalContext context) {
        super(context);
    }

    abstract public void assign(long constValue);

    abstract public void assign(int constValue);

    abstract public void assign(double constValue);

    abstract public void assign(float constValue);

    abstract public Object getDataObject();

    abstract public Class<? extends Number> getNumericType();

    abstract DoubleBuffer getPackedArray(DoubleBuffer buf);

    abstract FloatBuffer getPackedArray(FloatBuffer buf);

    abstract IntBuffer getPackedArray(IntBuffer buf);

    abstract void releasePackedArray(DoubleBuffer buf);

    abstract void releasePackedArray(FloatBuffer buf);

    abstract void releasePackedArray(IntBuffer buf);
}
/** @} */
