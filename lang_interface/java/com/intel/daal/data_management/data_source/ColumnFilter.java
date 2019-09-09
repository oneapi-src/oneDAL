/* file: ColumnFilter.java */
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
 * @defgroup data_sources Data Sources
 * @brief Specifies methods to access data
 * @ingroup data_management
 * @{
 */
/**
 */
package com.intel.daal.data_management.data_source;

import com.intel.daal.utils.*;
import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA_SOURCE__COLUMNFILTER"></a>
 */
public class ColumnFilter extends ModifierIface {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public ColumnFilter(DaalContext context) {
        super(context);
        this.cObject = cInit();
    }

    public ColumnFilter odd() {
        cOdd(cObject);
        return this;
    }

    public ColumnFilter even() {
        cEven(cObject);
        return this;
    }

    public ColumnFilter none() {
        cNone(cObject);
        return this;
    }

    public ColumnFilter list(long[] valid) {
        cList(cObject, valid);
        return this;
    }

    private native long cInit();
    private native void cOdd(long cObject);
    private native void cEven(long cObject);
    private native void cNone(long cObject);
    private native void cList(long cObject, long[] valid);
}
/** @} */
