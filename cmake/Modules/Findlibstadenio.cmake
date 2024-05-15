###############################################################################
# Find Staden IOLib
#
# This sets the following variables:
# STADEN_FOUND - True if Staden IOLib was found.
# STADEN_INCLUDE_DIR - Header files.
# STADEN_LIBRARIES - Staden IOLib library.

find_path(STADEN_INCLUDE_DIR io_lib
	HINTS ${STADEN_ROOT} ENV STADEN_ROOT
  PATH_SUFFIXES include)

find_library(STADEN_LIBRARY NAMES staden-read libstaden-read 
  HINTS ${STADEN_ROOT} ENV STADEN_ROOT PATH_SUFFIXES lib lib64)

find_library(HTSCODEC_LIBRARY NAMES htscodecs libhtscodecs
  HINTS ${STADEN_ROOT} ENV STADEN_ROOT PATH_SUFFIXES lib lib64)

if(STADEN_INCLUDE_DIR)
  set(_version_regex "^#define[ \t]+PACKAGE_VERSION[ \t]+\"([^\"]+)\".*")
  string(REGEX REPLACE "${_version_regex}" "\\1"
    STADEN_VERSION "${STADEN_VERSION}")
  unset(_version_regex)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(libstadenio DEFAULT_MSG
                                  STADEN_LIBRARY 
                                  STADEN_INCLUDE_DIR
                                  )

if (LIBSTADENIO_FOUND)
  message(STATUS "Staden IOLib found (include: ${STADEN_INCLUDE_DIR})")
  set(STADEN_LIBRARIES "${STADEN_LIBRARY};${HTSCODEC_LIBRARY}")
endif()

mark_as_advanced(STADEN_INCLUDE_DIR STADEN_LIBRARY)
