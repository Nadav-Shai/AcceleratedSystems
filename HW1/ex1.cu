#include "ex1.h"

__device__ void prefix_sum(int arr[], int arr_size) {
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

__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) {
    // TODO
    int tid = threadIdx.x;

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
                
                uchar value = all_in[flat_index];

                atomicAdd(&histogram[value], 1);
            }
            __syncthreads();

            prefix_sum(histogram, 256);

            for (int v = tid; v < 256; v += blockDim.x) {
                int maps_true_index = (tile_row * TILE_COUNT + tile_col) * 256 + v;
                maps[maps_true_index] = (uchar)((histogram[v] * 255) / (TILE_WIDTH * TILE_WIDTH));
            }
            __syncthreads();
        }
    }

    interpolate_device(maps, all_in, all_out);
    return; 
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    // TODO define task serial memory buffers
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    //TODO: allocate GPU memory for a single input image, a single output image, and maps

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: in a for loop:
    //   1. copy the relevant image from images_in to the GPU memory you allocated
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init

    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for all the input images, output images, and maps

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init

    free(context);
}
