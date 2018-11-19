/* file: LibUtils.java */
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
 * @brief Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) package
 */
package com.intel.daal.utils;

import java.io.*;
import java.util.Date;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * @ingroup libraryUtilities
 * @{
 */
/**
 * <a name="DAAL-CLASS-LIBUTILS"></a>
 */
public final class LibUtils{
    private static final String LIBRARY_PATH_IN_JAR = "/lib";
    private final static String DAALLIB      = "JavaAPI";
    private final static String TBBLIB       = "tbb";
    private final static String TBBMALLOCLIB = "tbbmalloc";

    private final static String subDir = "daal_" + new Date().getTime();

    private static final Logger logger = Logger.getLogger(LibUtils.class.getName());
    private static final Level logLevel = Level.FINE;

    /**
     * Load JavaAPI DAAL lib and TBB libs
     */
    public static void loadLibrary()
    {
        try {
            logger.log(logLevel, "Loading library " + DAALLIB + " as file.");
            System.loadLibrary(DAALLIB);
            logger.log(logLevel, "DONE: Loading library " + DAALLIB + " as file.");
            return;
        }
        catch (UnsatisfiedLinkError e) {
            logger.log(logLevel, "Can`t find library " + DAALLIB + " in java.library.path.");
        }

        try {
            loadFromJar(subDir, TBBLIB);
            loadFromJar(subDir, TBBMALLOCLIB);
            loadFromJar(subDir, DAALLIB);
            return;
        }
        catch (Throwable e) {
            logger.log(logLevel, "Error: Can`t load library as resource. " + e);
        }
    }

    /**
     * Load lib as resource
     * @param path   sub folder (in temporary folder) name
     * @param name   library name
     */
    private static void loadFromJar(String path, String name) throws Throwable
    {
        String FullName = createLibraryFileName(name);

        File fileOut = createTempFile(path, FullName);
        if (fileOut == null) {
            logger.log(logLevel, "DONE: Loading library " + FullName + " as resource.");
            return;
        }

        InputStream streamIn = LibUtils.class.getResourceAsStream(LIBRARY_PATH_IN_JAR + "/" + FullName);
        if (streamIn == null)
        {
            throw new IOException("Error: No resource " + LIBRARY_PATH_IN_JAR + "/" + FullName + " found.");
        }

        try(OutputStream streamOut = new FileOutputStream(fileOut))
        {
            logger.log(logLevel, "Writing resource " + LIBRARY_PATH_IN_JAR + "/" + FullName + " to temp file " + fileOut.getAbsolutePath());

            byte[] buffer = new byte[32768];
            while (true)
            {
                int read = streamIn.read(buffer);
                if (read < 0)
                {
                    break;
                }
                streamOut.write(buffer, 0, read);
            }

            streamOut.flush();
        }
        catch (IOException e)
        {
            throw new IOException("Error:  I/O error occurs from/to temp file " + fileOut.getAbsolutePath());
        }
        finally
        {
            streamIn.close();
        }

        System.load(fileOut.toString());
        logger.log(logLevel, "DONE: Loading library " + FullName + " as resource.");
    }

    /**
     * Construct library file name
     * @param name   library name
     *
     * @return constructed file name
     */
    public static String createLibraryFileName(String name) throws IOException
    {
        String fullName = name;

        String OSname = System.getProperty("os.name");
        OSname = OSname.toLowerCase();

        if (OSname.startsWith("windows")) {
            fullName = name + ".dll";
            return fullName;
        }

        if (OSname.startsWith("linux")) {
            if (name.contains("tbb")) {
                fullName = "lib" + name + ".so.2";
            }
            else {
                fullName = "lib" + name + ".so";
            }
            return fullName;
        }

        if (OSname.startsWith("mac os")) {
            fullName = "lib" + name + ".dylib";
            return fullName;
        }

        throw new IOException("Error: Unknown OS " + OSname );
    }

    /**
     * Create temporary file
     * @param name   library name
     * @param tempSubDirName   sub folder (in temporary folder) name
     *
     * @return temporary file handler. null if file exist already.
     */
    private static File createTempFile(String tempSubDirName, String name) throws IOException
    {
        File tempSubDirectory = new File(System.getProperty("java.io.tmpdir") + "/" + tempSubDirName + LIBRARY_PATH_IN_JAR);

        if (!tempSubDirectory.exists())
        {
            boolean createdDirectory = tempSubDirectory.mkdirs();
            if (!createdDirectory)
            {
                throw new IOException("Error: Can`t create folder for temp file " + tempSubDirectory);
            }
        }

        String tempFileName = tempSubDirectory + "/" + name;
        File tempFile = new File(tempFileName);

        if (tempFile == null)
        {
            throw new IOException("Error: Can`t create temp file " + tempFile);
        }

        if (tempFile.exists())
        {
            return null;
        }

        return tempFile;
    }

}
/** @} */
