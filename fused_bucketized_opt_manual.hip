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
  __device__ int operator()(const float value, const float* boundaries, int count) {
    int bucket = 0;
    while (count > 0) {
      int step = count >> 1;
      int left = bucket + step;
      int cond = (value >= boundaries[left]);
      bucket = cond * (left + 1) + (1 - cond) * bucket;
      count = cond * (count - step - 1) + (1 - cond) * step;
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
  cudaMemsetAsync(dst, byte, total_bytes, stream);
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

template <typename A, typename B, typename C, typename Factory>
__global__ void fused_element_wise_kernel(const A** a, const B** b, C** c,
                                          int64_t N, int64_t* sizes,
                                          int64_t* b_sizes,
                                          Factory factory) {
  int64_t vec_id = blockIdx.y;
  int64_t size_local = sizes[vec_id];
  int64_t b_size_local = b_sizes[vec_id];
  int64_t threads_num = blockDim.x * gridDim.x;
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ A sm_input[];
  extern __shared__ B sm_boundaries[];

  // move data to shared buffer
  for (int64_t idx = threadIdx.x; idx < b_size_local; idx += blockDim.x) {
    sm_boundaries[idx] = b[vec_id][idx];
  }
  __syncthreads();

  for (int64_t index = tid; index < size_local; index += threads_num) {
    c[vec_id][index] = factory(a[vec_id][index], sm_boundaries, b_size_local);
  }
}

template <typename A, typename B, typename C, typename Factory>
void fused_element_wise_launcher(const A** a, const B** b, C** c, int64_t* sizes,
                                 int64_t* b_sizes, int64_t N, Factory factor,
                                 bool with_pack, hipStream_t stream) {
  int64_t sm_count = get_sm_count();
  int64_t max_size = 0;
  std::vector<int64_t> offsets(N + 1, 0);
  for (int64_t i = 0; i < N; ++i) {
    max_size = std::max(max_size, sizes[i]);
  }

  int64_t block_num =
      min(sm_count * 8, (max_size + KBLOCK_SIZE - 1) / KBLOCK_SIZE);
  // std::cout << "block_num = " << block_num << std::endl;
  dim3 grid(block_num, N);
  dim3 block(KBLOCK_SIZE);
  int64_t* d_sizes = cuda_malloc_and_copy<int64_t>(sizes, N, stream);
  int64_t* b_d_sizes = cuda_malloc_and_copy<int64_t>(b_sizes, N, stream);

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
    HIP_CHECK(hipEventRecord(start, hipStreamDefault));
    fused_element_wise_kernel<A, B, C, Factory>
          <<<grid, block, b_sizes[0] * sizeof(B), stream>>>(
                                                      const_cast<const A**>(d_a),
                                                      const_cast<const B**>(d_b),
                                                      d_c, N, d_sizes, b_d_sizes,
                                                      factor);

    HIP_CHECK(hipEventRecord(stop, hipStreamDefault)); 
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

  // int64_t sm_total_size = b_sizes[0] * sizeof(B) + KBLOCK_SIZE * sizeof(A);
  // fused_element_wise_kernel<A, B, C, Factory>
  //       <<<grid, block, sm_total_size, stream>>>(const_cast<const A**>(d_a),
  //                                                const_cast<const B**>(d_b),
  //                                                d_c, N, d_sizes, b_d_sizes,
  //                                                factor);

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
  for (int i = 0; i < shapes.size(); ++i) {
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
  for (int i = 0; i < shapes.size(); ++i) {
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
  for (int64_t i = 0; i < shapes.size(); ++i) {
    d_outputs_ptr.push_back((int64_t*)malloc(shapes[i] * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_outputs_ptr[i], outputs[i].data(),
                        shapes[i] * sizeof(int64_t), hipMemcpyDeviceToHost));
  }

  // call cpu
  std::vector<CustomTensor<float>> cpu_values;
  std::vector<float*> h_value_ptrs;
  for (int i = 0; i < shapes.size(); ++i) {
    h_value_ptrs.push_back((float*)malloc(shapes[i] * sizeof(float)));
    HIP_CHECK(hipMemcpy(h_value_ptrs[i], values[i].data(),
                        shapes[i] * sizeof(float), hipMemcpyDeviceToHost));
    cpu_values.emplace_back(CustomTensor<float>({shapes[i]}, h_value_ptrs[i]));
  }

  std::vector<CustomTensor<float>> cpu_boundaries;
  for (int i = 0; i < shapes.size(); ++i) {
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
  for (int i = 0; i < shapes.size(); ++i) {
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
