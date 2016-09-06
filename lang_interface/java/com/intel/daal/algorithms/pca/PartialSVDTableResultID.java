/* file: PartialSVDTableResultID.java */
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

package com.intel.daal.algorithms.pca;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__PARTIALSVDTABLERESULTID"></a>
 * @brief Available identifiers of partial results of the %SVD method of the PCA algorithm
 */
public final class PartialSVDTableResultID {
    private int _value;

    static {
        System.loadLibrary("JavaAPI");
    }

    public PartialSVDTableResultID(int value) {
        _value = value;
    }

    /** Returns value of input identifier
      * \return value of input identifier */
    public int getValue() {
        return _value;
    }

    private static final int nObservationsId = 0;
    private static final int sumSVDId        = 1;
    private static final int sumSquaresSVDId = 2;

    /*!< Number of observations */
    public static final PartialSVDTableResultID nObservations = new PartialSVDTableResultID(nObservationsId);
    /*!< Array of sums */
    public static final PartialSVDTableResultID sumSVD        = new PartialSVDTableResultID(sumSVDId);
    /*!< Array of sums of squares */
    public static final PartialSVDTableResultID sumSquaresSVD = new PartialSVDTableResultID(sumSquaresSVDId);
}
