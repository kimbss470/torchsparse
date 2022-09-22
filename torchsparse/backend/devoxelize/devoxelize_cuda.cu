#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <torch/extension.h>

#include <THC/THCAtomics.cuh>

#include "nvToolsExt.h"


template <typename scalar_t>
__global__ void calc_ti_weights_kernel(int N, const float scale, const scalar_t *__restrict__ coords, const int *__restrict__ indices, scalar_t *__restrict__ weight) {


    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        int idx_x = index * 3;
        int idx_y = index * 3 + 1;
        int idx_z = index * 3 + 2;
        scalar_t x = coords[idx_x];
        scalar_t y = coords[idx_y];
        scalar_t z = coords[idx_z];

        scalar_t xf = (int)(x / scale) * scale;
        scalar_t yf = (int)(y / scale) * scale;
        scalar_t zf = (int)(z / scale) * scale;
    
        scalar_t xc = xf + scale; 
        scalar_t yc = yf + scale; 
        scalar_t zc = zf + scale; 

        scalar_t w0 = (xc - x) * (yc - y) * (zc - z);
        scalar_t w1 = (xc - x) * (yc - y) * (z - zf);
        scalar_t w2 = (xc - x) * (y - yf) * (zc - z);
        scalar_t w3 = (xc - x) * (y - yf) * (z - zf);
        scalar_t w4 = (x - xf) * (yc - y) * (zc - z);
        scalar_t w5 = (x - xf) * (yc - y) * (z - zf);
        scalar_t w6 = (x - xf) * (y - yf) * (zc - z);
        scalar_t w7 = (x - xf) * (y - yf) * (z - zf);
        
        w0 = (indices[index] == -1) ? (scalar_t)0 : w0;
        w1 = (indices[index+1] == -1) ? (scalar_t)0 : w1;
        w2 = (indices[index+2] == -1) ? (scalar_t)0 : w2;
        w3 = (indices[index+3] == -1) ? (scalar_t)0 : w3;
        w4 = (indices[index+4] == -1) ? (scalar_t)0 : w4;
        w5 = (indices[index+5] == -1) ? (scalar_t)0 : w5;
        w6 = (indices[index+6] == -1) ? (scalar_t)0 : w6;
        w7 = (indices[index+7] == -1) ? (scalar_t)0 : w7;

        scalar_t sum_w = w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7 + 1e-8;
        w0 /= sum_w; 
        w1 /= sum_w;
        w2 /= sum_w;
        w3 /= sum_w;
        w4 /= sum_w;
        w5 /= sum_w;
        w6 /= sum_w;
        w7 /= sum_w;

        weight[index*8] = w0;
        weight[index*8+1] = w1;
        weight[index*8+2] = w2;
        weight[index*8+3] = w3;
        weight[index*8+4] = w4;
        weight[index*8+5] = w5;
        weight[index*8+6] = w6;
        weight[index*8+7] = w7;
    }

} 

// input features (n, c), indices (N, 8), weight (N, 8) -> output features (N,
// c)
template <typename scalar_t>
__global__ void devoxelize_forward_kernel(int N, int c,
                                          const int *__restrict__ indices,
                                          const scalar_t *__restrict__ weight,
                                          const scalar_t *__restrict__ feat,
                                          scalar_t *__restrict__ out) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i = index / c;
  int j = index % c;

  if (i < N) {
    const int *indices_ = indices + 8 * i;
    const scalar_t *weight_ = weight + 8 * i;
    const scalar_t *feat_ = feat + j;

    scalar_t cur_feat;
    for (int k = 0; k < 8; k++) {
      cur_feat = 0;
      if (indices_[k] >= 0) cur_feat = feat_[indices_[k] * c];

      out[i * c + j] += weight_[k] * cur_feat;
    }
  }
}

// input weight (N, 8), indices (N, 8), top_grad (N, c) -> bottom grad (n, c)
template <typename scalar_t>
__global__ void devoxelize_backward_kernel(
    int N, int n, int c, const int *__restrict__ indices,
    const scalar_t *__restrict__ weight, const scalar_t *__restrict__ top_grad,
    scalar_t *__restrict__ bottom_grad) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i = index / c;
  int j = index % c;

  if (i < N) {
    const int *indices_ = indices + 8 * i;
    const scalar_t *weight_ = weight + 8 * i;

    scalar_t cur_top_grad = top_grad[i * c + j];

#pragma unroll
    for (int k = 0; k < 8; k++) {
      if (indices_[k] >= 0)
        atomicAdd(&bottom_grad[indices_[k] * c + j], weight_[k] * cur_top_grad);
    }
  }
}


at::Tensor calc_ti_weights_cuda(const at::Tensor coords, 
                                const at::Tensor indices,
                                float scale ) {
    int N = coords.size(0);
    at::Tensor weight = torch::zeros({N,8}, at::device(coords.device()).dtype(coords.dtype()));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        coords.type(), "calc_ti_weights_cuda", ([&] {
            calc_ti_weights_kernel<scalar_t><<<N/64, 64>>>(
                N, scale, coords.data_ptr<scalar_t>(), indices.data_ptr<int>(), weight.data_ptr<scalar_t>());

        }));
    return weight;
}
// make sure indices is int type
// feat: (b,c,s) indices: (N, 3) batch_index: (N, ) -> out: (N, c)
at::Tensor devoxelize_forward_cuda(const at::Tensor feat,
                                   const at::Tensor indices,
                                   const at::Tensor weight) {
  int c = feat.size(1);
  int N = indices.size(0);

  nvtxRangePushA("cuda devoxelize");
  at::Tensor out =
      torch::zeros({N, c}, at::device(feat.device()).dtype(feat.dtype()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      feat.type(), "devoxelize_forward_cuda", ([&] {
        devoxelize_forward_kernel<scalar_t><<<N, c>>>(
            N, c, indices.data_ptr<int>(), weight.data_ptr<scalar_t>(),
            feat.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
      }));
  nvtxRangePop();
  return out;
}

// top_grad: (N, c), indices: (N, 3), batch_index: (N, ) -> bottom_grad:
// (b,c,s), s=r^3
at::Tensor devoxelize_backward_cuda(const at::Tensor top_grad,
                                    const at::Tensor indices,
                                    const at::Tensor weight, int n) {
  int c = top_grad.size(1);
  int N = top_grad.size(0);
  at::Tensor bottom_grad = torch::zeros(
      {n, c}, at::device(top_grad.device()).dtype(top_grad.dtype()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "devoxelize_backward_cuda", ([&] {
        devoxelize_backward_kernel<scalar_t><<<N, c>>>(
            N, n, c, indices.data_ptr<int>(), weight.data_ptr<scalar_t>(),
            top_grad.data_ptr<scalar_t>(), bottom_grad.data_ptr<scalar_t>());
      }));

  return bottom_grad;
}
