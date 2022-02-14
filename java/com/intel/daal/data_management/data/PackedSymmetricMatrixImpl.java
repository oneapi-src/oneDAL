/* file: PackedSymmetricMatrixImpl.java */
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
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import com.intel.daal.services.DaalContext;

abstract class PackedSymmetricMatrixImpl extends NumericTableImpl {
    protected Class<? extends Number> type;

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the packed symmetric matrix
     * @param context   Context to manage the packed symmetric matrix
     */
    public PackedSymmetricMatrixImpl(DaalContext context) {
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
