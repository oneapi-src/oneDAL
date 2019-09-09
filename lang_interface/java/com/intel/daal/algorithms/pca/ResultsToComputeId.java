/* file: ResultsToComputeId.java */
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
 * @ingroup pca
 * @{
 */
package com.intel.daal.algorithms.pca;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__RESULTTOCOMPUTEID"></a>
 * \brief Available identifiers of results of the PCA algorithm
 */
public final class ResultsToComputeId {

    public static final long none     = 0x0000000000000000L; /*!< No optional result */
    public static final long mean     = 0x0000000000000001L; /*!< Compute mean */
    public static final long variance = 0x0000000000000002L; /*!< Compute variance */
    public static final long eigenvalue = 0x0000000000000004L; /*!< Compute variance */
}/** @} */
