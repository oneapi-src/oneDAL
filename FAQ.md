# Frequently Asked Questions
This document provides answers to frequently asked questions about Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL).
Let us know if you have a question that is not covered here via [https://github.com/01org/daal/issues](https://github.com/01org/daal/issues) or [Intel(R) DAAL Forum](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library).

## Building Intel(R) DAAL from sources

### Windows*

* During the build process I can see errors reported by `find` util. Any ideas on what can cause them?

It is likely caused by execution of MSYS2* `find` command on FAT32 file system. Building Intel DAAL on NTFS should not cause these issues

* What can cause Makefile fail with "fork: Resource temporarily unavailable" error?

This error can be caused by executing MSYS2* `make` util with large number of threads. You can reduce number of threads to solve this issue. 
Try using make with smaller number of threads, e.g. `make daal -j 4 PLAT=win32e`
