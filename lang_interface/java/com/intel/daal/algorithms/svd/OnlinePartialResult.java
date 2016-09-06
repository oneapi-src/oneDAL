/* file: OnlinePartialResult.java */
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

package com.intel.daal.algorithms.svd;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__ONLINEPARTIALRESULT"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of the SVD algorithm in the online processing or
 * distributed processing modes
*/
public class OnlinePartialResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public OnlinePartialResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns partial results of the algorithm
     * @param id    Identifier of the parameter
     * @return      Parameter that corresponds to the given identifier
     */
    public DataCollection get(PartialResultId id) {
        if (id == PartialResultId.outputOfStep1ForStep3) {
            return new DataCollection(getContext(),
                    cGetDataCollection(getCObject(), PartialResultId.outputOfStep1ForStep3.getValue()));
        } else if (id == PartialResultId.outputOfStep1ForStep2) {
            return new DataCollection(getContext(),
                    cGetDataCollection(getCObject(), PartialResultId.outputOfStep1ForStep2.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cGetDataCollection(long cOnlinePartialResult, int id);
}
