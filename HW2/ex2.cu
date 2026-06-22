#include "ex2.h"
#include <cuda/atomic>

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
// TODO implement a MPMC queue
// TODO implement the persistent kernel
// TODO implement a function for calculating the threadblocks count

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
public:
    queue_server(int threads)
    {
        // TODO initialize host state
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        return false;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        return false;

        // TODO return the img_id of the request that was completed.
        //*img_id = ... 
        return true;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
