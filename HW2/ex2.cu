#include "ex2.h"
#include <cuda/atomic>

#define SHMEM_USAGE 2048
#define REGS_PER_THREAD 32

__device__ void prefix_sum(int arr[], int arr_size) {
    // TODO complete according to hw1
    int tid = threadIdx.x;
    int increment;
    for (int stride = 1; stride < arr_size; stride *= 2) {
        if (tid < arr_size && tid >= stride) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid < arr_size && tid >= stride) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__
 void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

__device__
void process_image(uchar *in, uchar *out, uchar* maps) {
    // TODO complete according to hw1
    int tid = threadIdx.x;
    int img_idx = blockIdx.x;

    uchar* in_img = in + img_idx * IMG_WIDTH * IMG_HEIGHT;
    uchar* out_img = out + img_idx * IMG_WIDTH * IMG_HEIGHT;
    uchar* img_maps = maps + img_idx * TILE_COUNT * TILE_COUNT * 256;
    
    __shared__ int histogram[256];

    for (int tile_row = 0; tile_row < TILE_COUNT; tile_row++) {
        for (int tile_col = 0; tile_col < TILE_COUNT; tile_col++) {
            for (int i = tid; i < 256; i += blockDim.x) {
                histogram[i] = 0;
            }
            __syncthreads();

            for (int i = tid; i < TILE_WIDTH * TILE_WIDTH; i += blockDim.x) { 
                int local_row = i / TILE_WIDTH;
                int local_col = i % TILE_WIDTH;

                int global_row = tile_row * TILE_WIDTH + local_row;
                int global_col = tile_col * TILE_WIDTH + local_col;

                int flat_index = global_row * IMG_WIDTH + global_col;
                
                uchar value = in_img[flat_index];

                atomicAdd(&histogram[value], 1);
            }
            __syncthreads();

            prefix_sum(histogram, 256);

            for (int v = tid; v < 256; v += blockDim.x) {
                int maps_true_index = (tile_row * TILE_COUNT + tile_col) * 256 + v;
                img_maps[maps_true_index] = (uchar)((histogram[v] * 255) / (TILE_WIDTH * TILE_WIDTH));
            }
            __syncthreads();
        }
    }

    interpolate_device(img_maps, in_img, out_img);
    return;
}

__global__
void process_image_kernel(uchar *in, uchar *out, uchar* maps){
    process_image(in, out, maps);
}

class streams_server : public image_processing_server
{
private:
    // TODO define stream server context (memory buffers, streams, etc...)
    cudaStream_t streams[STREAM_COUNT];
    uchar* in[STREAM_COUNT];
    uchar* out[STREAM_COUNT];
    uchar* maps[STREAM_COUNT];
    int img_ids[STREAM_COUNT];
    bool busy[STREAM_COUNT];
    // uchar* host_out[STREAM_COUNT];

    const size_t img_size = IMG_HEIGHT * IMG_WIDTH * sizeof(uchar);
    const size_t maps_size =  TILE_COUNT * TILE_COUNT * 256 * sizeof(uchar);
        
public:
    streams_server()
    {
        // TODO initialize context (memory buffers, streams, etc...)
        
        for (int i = 0; i < STREAM_COUNT; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            CUDA_CHECK(cudaMalloc(&in[i], img_size));
            CUDA_CHECK(cudaMalloc(&out[i], img_size));
            CUDA_CHECK(cudaMalloc(&maps[i], maps_size));
            busy[i] = false;
        }
    }

    ~streams_server() override
    {
        // TODO free resources allocated in constructor
        for (int i = 0; i < STREAM_COUNT; i++) {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
            CUDA_CHECK(cudaFree(in[i]));
            CUDA_CHECK(cudaFree(out[i]));
            CUDA_CHECK(cudaFree(maps[i]));
        }
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO place memory transfers and kernel invocation in streams if possible.
        int threads_per_block = 1024;
        for (int i = 0; i < STREAM_COUNT; i++) {
            if (busy[i]) continue;
            img_ids[i] = img_id;
            busy[i] = true;
            cudaMemcpyAsync(in[i], img_in, img_size, 
                            cudaMemcpyHostToDevice, streams[i]);
            process_image_kernel<<<1, threads_per_block, 0, streams[i]>>>
                                (in[i], out[i], maps[i]);
            cudaMemcpyAsync(img_out, out[i], img_size, 
                            cudaMemcpyDeviceToHost, streams[i]);
            return true;
            
        }
        return false;
    }

    bool dequeue(int *img_id) override
    {  
        // TODO query (don't block) streams for any completed requests.
        for (int i = 0; i < STREAM_COUNT; i++)
        {
            if (!busy[i]) continue;
            
            cudaError_t status = cudaStreamQuery(streams[i]); // TODO query diffrent stream each iteration
            switch (status) {
            case cudaSuccess:
                // TODO return the img_id of the request that was completed.
                *img_id = img_ids[i];
                busy[i] = false;
                return true;
            case cudaErrorNotReady:
                continue;
            default:
                CUDA_CHECK(status);
                return false;
            }
        }
        return false;
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

// TODO implement a lock
struct ttas_lock {
    cuda::atomic<int, cuda::thread_scope_device> flag;

    ttas_lock() : flag(0) {}

    __device__ void lock() {
        while (true) {
            while (flag.load(cuda::memory_order_relaxed) != 0);

            if (flag.exchange(1, cuda::memory_order_acquire) == 0)
                return;
        }
    }

    __device__ void unlock() {
        flag.store(0, cuda::memory_order_release);
    }

};

// TODO implement a MPMC queue
struct cpu_to_gpu_slot {
    int img_id;
    uchar* img_in;
    uchar* img_out;
};

struct gpu_to_cpu_slot {
    int img_id;
};

template<typename T>
struct mpmc_queue {
    T* slots;
    int size;
    cuda::atomic<int, cuda::thread_scope_system> head;
    cuda::atomic<int, cuda::thread_scope_system> tail;

    mpmc_queue(T* slots, int size) : slots(slots), size(size), head(0), tail(0) {}

    __host__ bool cpu_push(T item) {
        int t = tail.load(cuda::memory_order_relaxed);
        if (t - head.load(cuda::memory_order_acquire) >= size)
            return false;
        slots[t % size] = item;
        tail.store(t + 1, cuda::memory_order_release);
        return true;
    }

    __host__ bool cpu_pop(T& item) {
        int h = head.load(cuda::memory_order_relaxed);
        if (h >= tail.load(cuda::memory_order_acquire))
            return false; // empty
        item = slots[h % size];
        head.store(h + 1, cuda::memory_order_release);
        return true;
    }

    __device__ T gpu_pop(ttas_lock& lock) {
        while (true) {
            lock.lock();
            int h = head.load(cuda::memory_order_relaxed);
            if (h < tail.load(cuda::memory_order_acquire)) {
                T item = slots[h % size];
                head.store(h + 1, cuda::memory_order_release);
                lock.unlock();
                return item;
            }
            lock.unlock();
        }
        return T{}; // unreachable - only suppresses warning
    }

    __device__ void gpu_push(T item, ttas_lock& lock) {
        while (true) {
            lock.lock();
            int t = tail.load(cuda::memory_order_relaxed);
            if (t - head.load(cuda::memory_order_acquire) < size) {
                slots[t % size] = item;
                tail.store(t + 1, cuda::memory_order_release);
                lock.unlock();
                return;
            }
            lock.unlock();
        }
    }
};


// TODO implement the persistent kernel
__global__ void queue_kernel(
    mpmc_queue<cpu_to_gpu_slot>* requests,
    mpmc_queue<gpu_to_cpu_slot>* responses, 
    ttas_lock* req_lock,
    ttas_lock* resp_lock,
    uchar* in_bufs, // num_blocks * img_size
    uchar* out_bufs, // num_blocks * img_size
    uchar* maps_bufs // num blocks * maps_size
) {
    __shared__ cpu_to_gpu_slot req;
    int tid = threadIdx.x;
    size_t img_size = IMG_HEIGHT * IMG_WIDTH;

    while (true) {
        // Thread 0 dequeue, broadcast via shmem
        if (tid == 0) {
            req = requests->gpu_pop(*req_lock);
        }
        __syncthreads();

        // Check for termination
        if (req.img_id == -1) return;

        // Copy input pinned -> device
        uchar* my_in = in_bufs + blockIdx.x * img_size;
        for (int i = tid; i < (int)img_size; i += blockDim.x) {
            my_in[i] = req.img_in[i];
        }
        __syncthreads();
        
        // Process image
        process_image(in_bufs, out_bufs, maps_bufs);
        __syncthreads();

        //Copy output device -> pinned
        uchar* my_out = out_bufs + blockIdx.x * img_size;
        for (int i = tid; i < (int)img_size; i += blockDim.x) {
            req.img_out[i] = my_out[i]; 
        }
        __syncthreads();

        // Thread 0 enqueues response
        if (tid == 0 ) {
            gpu_to_cpu_slot resp = {req.img_id};
            responses->gpu_push(resp, *resp_lock);
        }
        __syncthreads();
    }
}

// TODO implement a function for calculating the threadblocks count
int calc_num_threadblocks(int threads_per_block) {
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int thread_limit = prop.maxThreadsPerMultiProcessor / threads_per_block;
    int shmem_limit = prop.sharedMemPerMultiprocessor / SHMEM_USAGE;
    int regs_limit = prop.regsPerMultiprocessor / (REGS_PER_THREAD * threads_per_block);

    int blocksPerSM = min(thread_limit, min(shmem_limit, regs_limit));

    return blocksPerSM * prop.multiProcessorCount;
}

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
    int num_blocks;
    int queue_size;

    // Queues in pinned host memory (GPU accesses head/tail atomics)
    mpmc_queue<cpu_to_gpu_slot>* requests;
    mpmc_queue<gpu_to_cpu_slot>* responses;

    // Slot arrays in pinned host memory (GPU reads/writes image pointers and ids)
    cpu_to_gpu_slot* request_slots;
    gpu_to_cpu_slot* response_slots;

    // Locks in GPU memory (GPU threads compete on these)
    ttas_lock* req_lock;
    ttas_lock* resp_lock;

    // Per-block device buffers
    uchar* in_bufs;
    uchar* out_bufs;
    uchar* maps_bufs;

public:
    queue_server(int threads)
    {
        // TODO initialize host state    
        num_blocks = calc_num_threadblocks(threads);
        queue_size = (int)pow(2, ceil(log2(16 * num_blocks)));
        
        CUDA_CHECK(cudaMallocHost(&request_slots, queue_size * sizeof(cpu_to_gpu_slot)));
        CUDA_CHECK(cudaMallocHost(&response_slots, queue_size * sizeof(gpu_to_cpu_slot)));

        CUDA_CHECK(cudaMallocHost(&requests,  sizeof(mpmc_queue<cpu_to_gpu_slot>)));
        requests  = new (requests)  mpmc_queue<cpu_to_gpu_slot>(request_slots,  queue_size);
        CUDA_CHECK(cudaMallocHost(&responses, sizeof(mpmc_queue<gpu_to_cpu_slot>)));
        responses = new (responses) mpmc_queue<gpu_to_cpu_slot>(response_slots, queue_size);

        CUDA_CHECK(cudaMalloc(&req_lock,  sizeof(ttas_lock)));
        CUDA_CHECK(cudaMemset(req_lock,  0, sizeof(ttas_lock)));
        CUDA_CHECK(cudaMalloc(&resp_lock, sizeof(ttas_lock)));
        CUDA_CHECK(cudaMemset(resp_lock, 0, sizeof(ttas_lock)));

        CUDA_CHECK(cudaMalloc(&in_bufs,   num_blocks * IMG_HEIGHT * IMG_WIDTH * sizeof(uchar)));
        CUDA_CHECK(cudaMalloc(&out_bufs,  num_blocks * IMG_HEIGHT * IMG_WIDTH * sizeof(uchar)));
        CUDA_CHECK(cudaMalloc(&maps_bufs, num_blocks * TILE_COUNT * TILE_COUNT * 256 * sizeof(uchar)));
        
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
        queue_kernel<<<num_blocks, threads>>>(
            requests, responses,
            req_lock, resp_lock,
            in_bufs, out_bufs, maps_bufs
        );
    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
        cpu_to_gpu_slot sentinel = {-1, nullptr, nullptr};
        for (int i = 0; i < num_blocks; i++) {
            while (!requests->cpu_push(sentinel));
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(maps_bufs));
        CUDA_CHECK(cudaFree(out_bufs));
        CUDA_CHECK(cudaFree(in_bufs));
        CUDA_CHECK(cudaFree(resp_lock));
        CUDA_CHECK(cudaFree(req_lock));
        responses->~mpmc_queue();
        CUDA_CHECK(cudaFreeHost(responses));
        requests->~mpmc_queue();
        CUDA_CHECK(cudaFreeHost(requests));
        CUDA_CHECK(cudaFreeHost(response_slots));
        CUDA_CHECK(cudaFreeHost(request_slots));
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        cpu_to_gpu_slot req = {img_id, img_in, img_out};
        return requests->cpu_push(req);
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.        
        // TODO return the img_id of the request that was completed.
        gpu_to_cpu_slot resp;
        if (responses->cpu_pop(resp)) {
            *img_id = resp.img_id;
            return true;
        }
        return false;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
