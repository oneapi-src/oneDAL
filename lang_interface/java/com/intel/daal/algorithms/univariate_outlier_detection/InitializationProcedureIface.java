/* file: InitializationProcedureIface.java */
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
 * @ingroup univariate_outlier_detection
 * @{
 */
package com.intel.daal.algorithms.univariate_outlier_detection;

import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__INITIALIZATIONPROCEDUREIFACE"></a>
 * @brief Abstract interface class for setting initial parameters of univariate outlier detection algorithm @DAAL_DEPRECATED
 */
@Deprecated
public abstract class InitializationProcedureIface extends SerializableBase {

    /**
     * Constructs the initialization procedure iface
     * @param context   Context to manage the algorithm
     */
    public InitializationProcedureIface(DaalContext context) {
        super(context);
    }

    /**
     * Sets initial parameters of univariate outlier detection algorithm
     * @param data        %Input data table of size n x p
     * @param location    Vector of mean estimates of size 1 x p
     * @param scatter     Measure of spread, the variance-covariance matrix of size 1 x p
     * @param threshold   Limit that defines the outlier region, the array of size 1 x p containing a non-negative number
     */
    abstract public void initialize(NumericTable data, NumericTable location, NumericTable scatter,
                                    NumericTable threshold);
}
/** @} */
