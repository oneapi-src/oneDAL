/* file: FeatureManager.java */
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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA_SOURCE__FEATUREMANAGER"></a>
 */
public class FeatureManager extends ContextClient {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    FeatureManager(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    public void setDelimiter(char delimiter) {
        cSetDelimiter(cObject, delimiter);
    }

    private native void cSetDelimiter(long cObject, char delimiter);

    public void addModifier(ModifierIface modifier) {
        cAddModifier(cObject, modifier.getCObject());
    }

    private native void cAddModifier(long cObject, long cModifier);

    public void dispose() {
        if (this.cObject != 0) {
            cDispose(this.cObject);
            this.cObject = 0;
        }
    }

    private native void cDispose(long cObject);

    protected long cObject;
}
/** @} */
