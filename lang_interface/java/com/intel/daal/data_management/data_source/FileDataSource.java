/* file: FileDataSource.java */
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
 * @ingroup data_sources
 * @{
 */
package com.intel.daal.data_management.data_source;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA_SOURCE__FILEDATASOURCE"></a>
 * @brief Specifies the methods for accessing the data stored in files
 */
public class FileDataSource extends DataSource {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     *Default constructor
     */
    public FileDataSource(DaalContext context, String filename) {
        super(context);

        cObject = cInit(filename);
        featureManager = new FeatureManager(context, cGetFeatureManager(cObject));
    }

    /**
     * Constructor
     */
    public FileDataSource(DaalContext context, String filename, DictionaryCreationFlag doDict,
            NumericTableAllocationFlag doNT) {
        super(context);

        cObject = cInit(filename);
        featureManager = new FeatureManager(context, cGetFeatureManager(cObject));
        if (doDict.ordinal() == DictionaryCreationFlag.DoDictionaryFromContext.ordinal()) {
            this.createDictionaryFromContext();
        }

        if (doNT.ordinal() == NumericTableAllocationFlag.DoAllocateNumericTable.ordinal()) {
            this.allocateNumericTable();
        }
    }

    protected native long cInit(String filename);

    private native long cGetFeatureManager(long cObject);
}
/** @} */
