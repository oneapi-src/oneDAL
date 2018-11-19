/* file: SerializationExample.java */
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

/*
 //  Content:
 //     Java example of numeric table serialization
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-SERIALIZATIONEXAMPLE">
 * @example SerializationExample.java
 */

package com.intel.daal.examples.serialization;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;

import com.intel.daal.services.DaalContext;

class SerializationExample {
    /* Input data set parameters */
    private static final String dataset  = "../data/batch/serialization.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException {
        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Retrieve the data from an input file */
        dataSource.loadDataBlock();

        /* Retrieve a numeric table */
        NumericTable dataTable = dataSource.getNumericTable();

        /* Print the original data */
        Service.printNumericTable("Data before serialization:", dataTable);

        /* Serialize the numeric table into a byte buffer */
        byte[] buffer = serializeNumericTable(dataTable);

        /* Deserialize the numeric table from the byte buffer */
        NumericTable restoredDataTable = deserializeNumericTable(buffer);

        /* Print the restored data */
        Service.printNumericTable("Data after deserialization:", restoredDataTable);

        context.dispose();
    }

    private static byte[] serializeNumericTable(NumericTable dataTable) throws IOException {
        /* Create an output stream to serialize the numeric table */
        ByteArrayOutputStream outputByteStream = new ByteArrayOutputStream();
        ObjectOutputStream outputStream = new ObjectOutputStream(outputByteStream);

        /* Serialize the numeric table into the output stream */
        dataTable.pack();
        outputStream.writeObject(dataTable);

        /* Store the serialized data in an array */
        byte[] buffer = outputByteStream.toByteArray();
        return buffer;
    }

    private static NumericTable deserializeNumericTable(byte[] buffer) throws IOException, ClassNotFoundException {
        /* Create an input stream to deserialize the numeric table from the array */
        ByteArrayInputStream inputByteStream = new ByteArrayInputStream(buffer);
        ObjectInputStream inputStream = new ObjectInputStream(inputByteStream);

        /* Create a numeric table object */
        NumericTable restoredDataTable = (NumericTable) inputStream.readObject();
        restoredDataTable.unpack(context);

        return restoredDataTable;
    }
}
