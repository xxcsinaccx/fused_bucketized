#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

#include <hip/hip_runtime.h>

constexpr int KBLOCK_SIZE = 256;
// static int free_time = 0;

#define HIP_CHECK(expr)                                            \
    do {                                                           \
        hipError_t err = expr;                                     \
        if (err != hipSuccess) {                                   \
            std::cerr << "HIP error at " << __FILE__ << ": "       \
                      << __LINE__ << ": "                          \
                      << hipGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    } while(0)

// struct BucketizeData {
//   float* boundaries;
//   int len;
//   BucketizeData() : boundaries(nullptr), len(0) {}
//   BucketizeData(float* boundaries, int len)
//       : boundaries(boundaries), len(len) {}
// };

template<typename T>
struct CustomTensor {
  std::vector<int64_t> dims;
  T* data_ptr;
  bool is_gpu_device = false;

  std::vector<int64_t> size() { return dims; }
  int64_t numel() { 
    return std::accumulate(dims.begin(), dims.end(), 1,
                            std::multiplies<int64_t>()); 
  }
  T* data() {
    return data_ptr;
  }

  CustomTensor() : dims(0), data_ptr(nullptr) {}
  CustomTensor(std::vector<int64_t> dims_, T* data_ptr_) : dims(dims_),
                data_ptr(data_ptr_) {}
  CustomTensor(std::vector<int64_t> dims_, T* data_ptr_, bool is_gpu_device_) : 
                dims(dims_), is_gpu_device(is_gpu_device_) {
    if (is_gpu_device_) {
      void* tmp_ptr = nullptr;
      HIP_CHECK(hipMalloc(&tmp_ptr, numel() * sizeof(T)));
      HIP_CHECK(hipMemcpy(tmp_ptr, data_ptr_, numel() * sizeof(T),
                            hipMemcpyHostToDevice));
      data_ptr = (T*)tmp_ptr;
    } else {
      data_ptr = data_ptr_;
    }
  }
  CustomTensor(const CustomTensor&) = delete;
  CustomTensor& operator=(const CustomTensor&) = delete;

  CustomTensor(CustomTensor&& other) noexcept {
      move_from(std::move(other));
  }

  CustomTensor& operator=(CustomTensor&& other) noexcept {
      if (this != &other) {
          release();
          move_from(std::move(other));
      }
      return *this;
  }

  ~CustomTensor() {
    release();
  }

  private:
    void release() {
        if (is_gpu_device && data_ptr != nullptr) {
            HIP_CHECK(hipFree(data_ptr));
            data_ptr = nullptr;
        }
    }

    void move_from(CustomTensor&& other) {
        dims = std::move(other.dims);
        data_ptr = other.data_ptr;
        is_gpu_device = other.is_gpu_device;

        other.data_ptr = nullptr;
        other.is_gpu_device = false;
    }
};

struct BucketizeFactory {
  __device__ __forceinline__ int operator()(const float value, const float* boundaries, int count) {
    // Fixed-iteration binary search to reduce loop control overhead
    int c = count;
    if (c <= 0) return 0;
    int bucket = 0;
    // ceil(log2(c)) for positive c
    int iters = 32 - __builtin_clz(c);
    #pragma unroll
    for (int i = 0; i < iters; ++i) {
      int step = c >> 1;
      int left = bucket + step;
      int cond = (value >= boundaries[left]);
      bucket = cond ? (left + 1) : bucket;
      c = cond ? (c - step - 1) : step;
    }
    return bucket;
  }
};

template<typename T>
void gen_data(std::vector<T>& out_values,
              const int& num=10,
              const int& min = 100,
              const int& max = 1000,
              const float& scale = 10.f) {
  std::random_device rd;
  std::mt19937 gen(rd());
  if constexpr (std::is_same<T, float>::value) {
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < num; ++i) {
      float r = dist(gen);
      out_values.push_back(r * scale);
    }
  }
  else if constexpr (std::is_same<T, int>::value) {
    std::uniform_int_distribution<int> dist(min, max);
    for (int i = 0; i < num; ++i) {
      float r = dist(gen);
      out_values.push_back(r);
    }
  } else {
    std::cerr << "Currently type is not supported!" << std::endl;
  }
}

__inline__ int get_sm_count() {
  int device;
  HIP_CHECK(hipGetDevice(&device));
  int sm_count;
  HIP_CHECK(hipDeviceGetAttribute(&sm_count,
                                  hipDeviceAttributeMultiprocessorCount,
                                  device));
  return sm_count;
}

template <typename T>
__inline__ T* cuda_malloc(size_t bytes, hipStream_t stream = 0) {
  if (bytes == 0) {
    return nullptr;
  }
  // auto allocator = c10::cuda::CUDACachingAllocator::get();
  // T* dst = reinterpret_cast<T*>(allocator->raw_allocate(bytes));
  // return dst;
  T* dst = nullptr;
  HIP_CHECK(hipMalloc(&dst, bytes));
  return dst;
}

template <typename T>
T* cuda_malloc_and_copy(T* src, int size, hipStream_t stream = 0,
                        bool async = true) {
  size_t total_bytes = size * sizeof(T);
  T* dst = cuda_malloc<T>(total_bytes, stream);
  HIP_CHECK(hipMemcpyAsync(dst, src, total_bytes, hipMemcpyHostToDevice,
                            stream));
  if (!async) {
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  return dst;
}

template <typename T>
T* cuda_malloc_and_memset(unsigned char byte, size_t size,
                          hipStream_t stream = 0, bool async = true) {
  size_t total_bytes = size * sizeof(T);
  T* dst = cuda_malloc<T>(total_bytes, stream);
  HIP_CHECK(hipMemsetAsync(dst, byte, total_bytes, stream));
  if (!async) {
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  return dst;
}

__inline__ void delete_cuda_ptr(void* ptr) {
  // auto allocator = c10::cuda::CUDACachingAllocator::get();
  // allocator->raw_delete(ptr);
  HIP_CHECK(hipFree(ptr));
}

// Optimized kernel: single extern shared block for boundaries, coalesced loads,
// unrolled grid-stride compute for ILP, restrict pointers, inlined factory.

template <typename A, typename B, typename C, typename Factory>
__global__ __launch_bounds__(KBLOCK_SIZE, 2) void fused_element_wise_kernel(const A** a, const B** b, C** c,
                                          int64_t N, int64_t* sizes,
                                          int64_t* b_sizes,
                                          Factory factory) {
  int64_t vec_id64 = blockIdx.y;
  if (vec_id64 >= N) return;
  // use 32-bit indices locally to reduce register pressure
  int vec_id = static_cast<int>(vec_id64);
  int size_local = static_cast<int>(sizes[vec_id]);
  int b_size_local = static_cast<int>(b_sizes[vec_id]);
  int threads_num = blockDim.x * gridDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ unsigned char shmem[];
  B* sm_boundaries = reinterpret_cast<B*>(shmem);

  // move data to shared buffer (only first b_size_local threads copy)
  const B* __restrict__ b_vec = b[vec_id];
  if (threadIdx.x < b_size_local) {
    sm_boundaries[threadIdx.x] = b_vec[threadIdx.x];
  }
  __syncthreads();

  const A* __restrict__ a_vec = a[vec_id];
  C* __restrict__ c_vec = c[vec_id];

  // Process multiple elements per thread to improve ILP and reduce loop overhead
  const int UNROLL = 8;
  for (int index = tid; index < size_local; index += threads_num * UNROLL) {
    int i0 = index;
    // software prefetch of a few elements in the unroll group
    A v0 = (i0 < size_local) ? a_vec[i0] : A(0);
    A v1 = ((i0 + threads_num) < size_local) ? a_vec[i0 + threads_num] : A(0);
    A v2 = ((i0 + 2 * threads_num) < size_local) ? a_vec[i0 + 2 * threads_num] : A(0);
    A v3 = ((i0 + 3 * threads_num) < size_local) ? a_vec[i0 + 3 * threads_num] : A(0);
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int ii = i0 + u * threads_num;
      if (ii < size_local) {
        A val = (u == 0) ? v0 : (u == 1) ? v1 : (u == 2) ? v2 : (u == 3) ? v3 : a_vec[ii];
        c_vec[ii] = factory(val, sm_boundaries, b_size_local);
      }
    }
  }
}


template <typename A, typename B, typename C, typename Factory>
void fused_element_wise_launcher(const A** a, const B** b, C** c, int64_t* sizes,
                                 int64_t* b_sizes, int64_t N, Factory factor,
                                 bool with_pack, hipStream_t stream) {
  (void)with_pack; // parameter retained to preserve interface; not used
  int64_t sm_count = get_sm_count();
  int64_t max_size = 0;
  std::vector<int64_t> offsets(N + 1, 0);
  for (int64_t i = 0; i < N; ++i) {
    max_size = std::max(max_size, sizes[i]);
  }

  // copy sizes to device
  int64_t* d_sizes = cuda_malloc_and_copy<int64_t>(sizes, N, stream);
  int64_t* b_d_sizes = cuda_malloc_and_copy<int64_t>(b_sizes, N, stream);

  // shared memory size based on maximum boundary size across tensors (+1 padding)
  int64_t max_b_size = 0;
  for (int64_t i = 0; i < N; ++i) {
    max_b_size = std::max(max_b_size, b_sizes[i]);
  }
  size_t shared_memory_size = static_cast<size_t>(max_b_size + 1) * sizeof(B);

  int maxActiveBlocksPerSM = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocksPerSM,
      fused_element_wise_kernel<A, B, C, Factory>,
      KBLOCK_SIZE,
      shared_memory_size));
  if (maxActiveBlocksPerSM <= 0) {
    maxActiveBlocksPerSM = 8; // fallback consistent with original heuristic
  }

  int64_t block_num = std::min<int64_t>(sm_count * maxActiveBlocksPerSM,
                                        (max_size + KBLOCK_SIZE - 1) / KBLOCK_SIZE);
  if (block_num <= 0) block_num = sm_count * maxActiveBlocksPerSM;
  // std::cout << "block_num = " << block_num << std::endl;
  dim3 grid(block_num, N);
  dim3 block(KBLOCK_SIZE);

  // copy cpu ptr to device ptr
  A** d_a;
  HIP_CHECK(hipMalloc(&d_a, N * sizeof(A*)));
  HIP_CHECK(hipMemcpy(d_a, a, N * sizeof(A*), hipMemcpyHostToDevice));
  B** d_b;
  HIP_CHECK(hipMalloc(&d_b, N * sizeof(B*)));
  HIP_CHECK(hipMemcpy(d_b, b, N * sizeof(B*), hipMemcpyHostToDevice));
  C** d_c;
  HIP_CHECK(hipMalloc(&d_c, N * sizeof(C*)));
  HIP_CHECK(hipMemcpy(d_c, c, N * sizeof(C*), hipMemcpyHostToDevice));

  // latency measurement
  double kernel_time = 0;
    // Create events to measure the execution time of the kernels.
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  const constexpr unsigned int iterations = 10;
  for(unsigned int i = 0; i < iterations; ++i)
  {
    float kernel_ms{};

    // Record the start event.
    HIP_CHECK(hipEventRecord(start, stream));
    fused_element_wise_kernel<A, B, C, Factory>
          <<<grid, block, shared_memory_size, stream>>>(
                                                      const_cast<const A**>(d_a),
                                                      const_cast<const B**>(d_b),
                                                      d_c, N, d_sizes, b_d_sizes,
                                                      factor);

    HIP_CHECK(hipEventRecord(stop, stream)); 
    HIP_CHECK(hipEventSynchronize(stop));

    // Get the execution time of the kernel and add it to the total count.
    HIP_CHECK(hipEventElapsedTime(&kernel_ms, start, stop));
    kernel_time += kernel_ms;
  }

  // Destroy hipEvents.
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  kernel_time /= iterations;

  std::cout << "The mean time needed for each iteration has been "
            << kernel_time << "ms" << std::endl;

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(stream));
  delete_cuda_ptr(d_sizes);
  delete_cuda_ptr(b_d_sizes);
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
}

void fused_bucketized_cuda(std::vector<CustomTensor<float>>& inputs,
                           std::vector<CustomTensor<int64_t>>& outputs,
                           std::vector<CustomTensor<float>>& boundaries) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  int64_t N = inputs.size();
  std::vector<int64_t> sizes(N);
  std::vector<int64_t> b_sizes(N);
  std::vector<float*> inputs_ptrs(N);
  std::vector<int64_t*> outputs_ptrs(N);
  std::vector<float*> boundaries_ptrs(N);

  for (int64_t i = 0; i < N; ++i) {
    sizes[i] = inputs[i].numel();
    inputs_ptrs[i] = inputs[i].data();
    outputs_ptrs[i] = outputs[i].data();
    b_sizes[i] = boundaries[i].numel();
    boundaries_ptrs[i] = boundaries[i].data();
  }

  fused_element_wise_launcher<float, float, int64_t, BucketizeFactory>(
                              const_cast<const float**>(inputs_ptrs.data()),
                              const_cast<const float**>(boundaries_ptrs.data()),
                              outputs_ptrs.data(), sizes.data(), b_sizes.data(),
                              N, BucketizeFactory(), false, stream);
}

int get_bucketized_value(const float value, CustomTensor<float>& data) {
  int bucket = 0;
  int count = data.numel();
  auto boundaries = data.data();
  while (count > 0) {
    int left = bucket;
    int step = count / 2;
    left += step;
    if (!(value < boundaries[left])) {
      bucket = ++left;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return bucket;
}

void fused_bucketized_cpu(std::vector<CustomTensor<float>>& inputs,
                           std::vector<CustomTensor<int64_t>>& outputs,
                           std::vector<CustomTensor<float>>& boundaries) {
  int64_t N = inputs.size();
  for (int64_t i = 0; i < N; ++i) {
    int64_t total_nums = inputs[i].numel();
    for (int j = 0; j < total_nums; ++j) {
      int bucket = get_bucketized_value(inputs[i].data()[j], boundaries[i]);
      outputs[i].data()[j] = bucket;
    }
  }
}

int main() {
  // constexpr int N = 3;
  constexpr int B = 10;

  std::vector<int> shapes = {1048576, 4194304, 16777216};
  // std::vector<int> shapes = {256};
  // std::vector<int> shapes;
  // gen_data<int>(shapes, N, 5, 10);
  
  std::vector<CustomTensor<float>> values;
  for (int i = 0; i < (int)shapes.size(); ++i) {
    std::vector<float> out_values;
    gen_data<float>(out_values, shapes[i]);
    values.emplace_back(CustomTensor<float>({shapes[i]}, out_values.data(),
                                            true));
  }

  std::vector<float> boundaries_data;
  for (int i = 1; i < B + 1; ++i) {
    boundaries_data.push_back(i);
  }

  std::vector<CustomTensor<float>> boundaries;
  for (int i = 0; i < (int)shapes.size(); ++i) {
    boundaries.emplace_back(CustomTensor<float>({B}, boundaries_data.data(),
                            true));
  }

  // construct output
  int64_t num_tensors = values.size();
  std::vector<int64_t> sizes(num_tensors);
  std::vector<CustomTensor<int64_t>> outputs;
  for (int64_t i = 0; i < num_tensors; ++i) {
    std::vector<int64_t> out_value(values[i].numel());
    outputs.emplace_back(CustomTensor<int64_t>({values[i].numel()},
                                                out_value.data(), true));
  }

  fused_bucketized_cuda(values, outputs, boundaries);
  HIP_CHECK(hipDeviceSynchronize());

  // copy back to cpu
  std::vector<int64_t*> d_outputs_ptr;
  for (int64_t i = 0; i < (int64_t)shapes.size(); ++i) {
    d_outputs_ptr.push_back((int64_t*)malloc(shapes[i] * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_outputs_ptr[i], outputs[i].data(),
                        shapes[i] * sizeof(int64_t), hipMemcpyDeviceToHost));
  }

  // call cpu
  std::vector<CustomTensor<float>> cpu_values;
  std::vector<float*> h_value_ptrs;
  for (int i = 0; i < (int)shapes.size(); ++i) {
    h_value_ptrs.push_back((float*)malloc(shapes[i] * sizeof(float)));
    HIP_CHECK(hipMemcpy(h_value_ptrs[i], values[i].data(),
                        shapes[i] * sizeof(float), hipMemcpyDeviceToHost));
    cpu_values.emplace_back(CustomTensor<float>({shapes[i]}, h_value_ptrs[i]));
  }

  std::vector<CustomTensor<float>> cpu_boundaries;
  for (int i = 0; i < (int)shapes.size(); ++i) {
    cpu_boundaries.emplace_back(CustomTensor<float>({B},
                                boundaries_data.data()));
  }

  // construct output
  std::vector<CustomTensor<int64_t>> cpu_outputs;
  std::vector<int64_t*> h_out_ptrs;
  for (int64_t i = 0; i < num_tensors; ++i) {
    h_out_ptrs.push_back((int64_t*)malloc(shapes[i] * sizeof(int64_t)));
    cpu_outputs.emplace_back(CustomTensor<int64_t>({values[i].numel()},
                                                    h_out_ptrs[i]));
  }

  fused_bucketized_cpu(cpu_values, cpu_outputs, cpu_boundaries);

  // check results
  bool is_pass = true;
  for (int i = 0; i < (int)shapes.size(); ++i) {
    for (int j = 0; j < shapes[i]; ++j) {
      if (d_outputs_ptr[i][j] != cpu_outputs[i].data()[j]) {
        std::cout << "The " << i << "th " << j << " element " << "cpu: "
                  << cpu_outputs[i].data()[j] << ", gpu: "
                  << d_outputs_ptr[i][j] << std::endl;
        is_pass = false;
        break;
      }
    }
  }

  for (auto ptr : h_value_ptrs) {
    if (ptr != nullptr) free(ptr);
  }
  for (auto ptr : d_outputs_ptr) {
    if (ptr != nullptr) free(ptr);
  }
  for (auto ptr : h_out_ptrs) {
    if (ptr != nullptr) free(ptr);
  }

  if (is_pass) {
    std::cout << "\n========================================================\n"
              << "======================== PASSED ========================\n"
              << "========================================================\n";
  } else {
    std::cout << "\n========================================================\n"
              << "======================== FAILED ========================\n"
              << "========================================================\n";

  }
}
