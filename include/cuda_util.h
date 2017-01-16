/// @file
////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Copyright (C) 2016/17      Christian Lessig, Otto-von-Guericke Universitaet Magdeburg
///
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  module     : Exercise 1
///
///  author     : lessig@isg.cs.ovgu.de
///
///  project    : GPU Programming
///
///  description: Cuda utility functions
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_

// includes, system
#include <iostream>

template< typename T >
void checkError(T result, char const *const func, const char *const file, int const line) {

  if (result)  {
      std::cerr << "CUDA error at " << file << "::" << line << " with error code "
                << static_cast<unsigned int>(result) << " for " << func << "()." << std::endl;
      cudaDeviceReset();
      exit(EXIT_FAILURE);
  }
}

#define checkErrorsCuda(val) checkError( (val), #val, __FILE__, __LINE__ )

#endif // _CUDA_UTIL_H_
