#include "gpunetwork.cuh"

__global__
void
render(int n_cols, int n_rows) {

int tid_x = blockIdx.x*blockDim.x+threadIdx.x;
int tid_y = blockIdx.y*blockDim.y+threadIdx.y;


}

bool
runDevice( const unsigned int n_rows, const unsigned int n_cols, Device::Input& in) {

  cudaDeviceProp device_props;
    gpuErrchk( cudaGetDeviceProperties(&device_props, 0));
  unsigned int max_threads_per_block = device_props.maxThreadsPerBlock;

  unsigned int max_threads_per_block_sqrt = std::sqrt( max_threads_per_block);
  assert( max_threads_per_block_sqrt * max_threads_per_block_sqrt == max_threads_per_block);
  dim3 num_threads_per_block( std::min( n_rows, max_threads_per_block_sqrt),
                              std::min( n_cols, max_threads_per_block_sqrt) );
  dim3 num_blocks( n_rows / num_threads_per_block.x, n_cols / num_threads_per_block.y);
  if( 0 == num_blocks.x) {
    num_blocks.x++;
  }
  if( 0 == num_blocks.y) {
    num_blocks.y++;
  }
  std::cout << "num_blocks = " << num_blocks.x << " / " << num_blocks.y << std::endl;
  std::cout << "num_threads_per_block = " << num_threads_per_block.x << " / "
                                          << num_threads_per_block.y << std::endl;

  cudaThreadSynchronize();

  for(int i = 0; i < 30; i++){
    render<<<num_blocks , num_threads_per_block>>>(n_cols, n_rows, in);
  }

  cudaThreadSynchronize();

  return true;
}
