// CODE ADAPTED FROM: https://leimao.github.io/blog/CUDA-Matrix-Multiplication/#Matrix-Multiplication-Optimizations
#include <iostream>
#include <vector>
#include <random>
#define BLOCK_DIM 32

/******************************************************************************/
/*                               Utilities                                    */
/******************************************************************************/
#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
std::vector<T> create_rand_vector(size_t n)
{
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_int_distribution<int> uniform_dist(-256, 256);

    std::vector<T> vec(n);
    for (size_t i{0}; i < n; ++i)
    {
        vec.at(i) = static_cast<T>(uniform_dist(e));
    }

    return vec;
}

template <typename T>
bool allclose(std::vector<T> const &vec_1, std::vector<T> const &vec_2,
              T const &abs_tol)
{
    if (vec_1.size() != vec_2.size())
    {
        return false;
    }
    for (size_t i{0}; i < vec_1.size(); ++i)
    {
        if (std::abs(vec_1.at(i) - vec_2.at(i)) > abs_tol)
        {
            printf("Elements and index %d do not match: (%f, %f)\n", i, vec_1.at(i), vec_2.at(i));
            return false;
        }
    }
    return true;
}

/******************************************************************************/
/*                              Main Functions                                */
/******************************************************************************/

// Computes C = A * B on the GPU
// A: M x K
// B: K x N
// C: M x N
template <typename T>
__global__ void mm_kernel(T const *A, T const *B, T *C, size_t M,
                          size_t K, size_t N)
{
    // 2D block and 2D thread
    // Each thread computes one cell in C.
    size_t i{blockIdx.y * blockDim.y + threadIdx.y};
    size_t j{blockIdx.x * blockDim.x + threadIdx.x};

    // Do not process outside the matrix.
    if ((i >= M) || (j >= N))
    {
        return;
    }

    // Dot product "reduction" over K dimension
    T acc_sum{0};
    for (size_t k{0}; k < K; ++k)
    {
        acc_sum += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] = acc_sum;
}

// Computes C = A * B on the CPU
// A: M x K
// B: K x N
// C: M x N
template <typename T>
void mm(T const *A, T const *B, T *C, size_t M, size_t K, size_t N)
{
    // Compute the cells in C sequentially.
    for (size_t i{0}; i < M; ++i)
    {
        for (size_t j{0}; j < N; ++j)
        {
            T acc_sum{0};
            for (size_t k{0}; k < K; ++k)
            {
                acc_sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc_sum;
        }
    }
}

/******************************************************************************/
/*                                  Driver                                    */
/******************************************************************************/

int main()
{
    // Problem size
    const size_t M{1024}, K{1024}, N{1024};

    // Create random data
    std::vector<float> const A_vec{create_rand_vector<float>(M * K)};
    std::vector<float> const B_vec{create_rand_vector<float>(K * N)};
    std::vector<float> C_CPU_vec(M * N);
    std::vector<float> C_GPU_vec(M * N);

    float const *A{A_vec.data()};
    float const *B{B_vec.data()};
    float *C_CPU{C_CPU_vec.data()};
    float *C_GPU{C_GPU_vec.data()};

    // Allocate device buffers
    float *d_A, *d_B, *d_C_GPU;
    checkCuda(cudaMalloc(&d_A, sizeof(float) * A_vec.size()));
    checkCuda(cudaMalloc(&d_B, sizeof(float) * B_vec.size()));
    checkCuda(cudaMalloc(&d_C_GPU, sizeof(float) * C_GPU_vec.size()));

    // Copy data from host to device.
    std::cout << "Copying data H->D..." << std::endl;
    checkCuda(cudaMemcpy(d_A, A, sizeof(float) * A_vec.size(),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, B, sizeof(float) * B_vec.size(),
                         cudaMemcpyHostToDevice));
    std::cout << "Done." << std::endl;

    // Configure threadblock and grid
    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid(1, 1);
    blocks_per_grid.x = std::ceil(static_cast<double>(N) /
                                  static_cast<double>(threads_per_block.x));
    blocks_per_grid.y = std::ceil(static_cast<double>(M) /
                                  static_cast<double>(threads_per_block.y));

    // Launch kernel!
    std::cout << "(Asynchronously) launching kernel!" << std::endl;
    mm_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C_GPU, M, K, N);

    // Synchronize and get errors
    cudaDeviceSynchronize();
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Matrix Multiplication kernel failed to execute."
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Copy data from device to host.
    std::cout << "Copying data D->H..." << std::endl;
    checkCuda(cudaMemcpy(C_GPU, d_C_GPU, sizeof(float) * C_GPU_vec.size(),
                         cudaMemcpyDeviceToHost));
    std::cout << "Done." << std::endl;

    // Free device buffer.
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C_GPU));

    /* Uncomment to check results! */
    // Run matmul on CPU
    // std::cout << "Running matmul on the CPU..." << std::endl;
    // mm(A, B, C_CPU, M, K, N);
    // std::cout << "Done." << std::endl;
    // if (allclose<float>(C_CPU_vec, C_GPU_vec, 1e-4))
    //     std::cout << "Results match!" << std::endl;
    // else
    //     std::cout << "Uh oh, results don't match!" << std::endl;

    return 0;
}