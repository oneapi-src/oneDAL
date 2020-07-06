## Backward compatibility test for the old library names
At some point, Intel(R) DAAL was renamed to oneDAL. To support backward
compatibility for the users who depends in Intel(R) DAAL, we provide symlinks to
the old names of the binaries on Linux* and MacOs*, and a copy of the interface
library on Windows*.

This directory contains smoke tests to check whether the user can build
application with the help of old command line.

### Linux* and MacOs*
Symlinks for both static and dynamic libraries:
```sh
libdaal_core.a        -> libonedal_core.a
libdaal_core.so       -> libonedal_core.so
libdaal_sequential.a  -> libonedal_sequential.a
libdaal_sequential.so -> libonedal_sequential.so
libdaal_thread.a      -> libonedal_thread.a
libdaal_thread.so     -> libonedal_thread.so
```

### Windows*
Copy of the core interface library for dynamic linking. There is no backward
compatibility mechanism for static libraries.
```sh
daal_core_dll.lib copy of onedal_core_dll.lib
```
