/* file: HomogenNumericTableImpl.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 * <a name="DAAL-CLASS-DATA__HOMOGENNUMERICTABLEIMPL__HOMOGENNUMERICTABLEIMPL"></a>
 * @brief A derivative class of the NumericTableImpl class, that provides common interfaces for
 *        different implementations of a homogen numeric table
 */
abstract class HomogenNumericTableImpl extends NumericTableImpl {
    protected Class<? extends Number> type;

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the homogen numeric table
     * @param context   Context to manage the homogen numeric table
     */
    public HomogenNumericTableImpl(DaalContext context) {
        super(context);
    }

    abstract public void assign(long constValue);

    abstract public void assign(int constValue);

    abstract public void assign(double constValue);

    abstract public void assign(float constValue);

    abstract public double[] getDoubleArray();

    abstract public float[] getFloatArray();

    abstract public long[] getLongArray();

    abstract public Object getDataObject();

    abstract public Class<? extends Number> getNumericType();

    abstract public void set(long row, long column, double value);

    abstract public void set(long row, long column, float value);

    abstract public void set(long row, long column, long value);

    abstract public void set(long row, long column, int value);

    abstract public double getDouble(long row, long column);

    abstract public float getFloat(long row, long column);

    abstract public long getLong(long row, long column);

    abstract public int getInt(long row, long column);
}
/** @} */
