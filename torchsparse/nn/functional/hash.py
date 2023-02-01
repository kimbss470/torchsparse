from typing import Optional

import torch

import torchsparse.backend

__all__ = ['sphash', 'cylinder_sphash']


#  def cylinder_hash(coords, voxel_size):
#
#      max_theta = int(360 / voxel_size[1])
#      N = coords.size(0)
#
#      out = torch.zeros(N, dtype=torch.long)
#      out[:] = 14695981039346656037 & 0xFFFFFFFFFFFFFFFF
#      for i in range(4):
#          torch.bitwise_xor(out, coords[:,:,i], out)
#          out = out * (1099511628211 & 0xFFFFFFFFFFFFFFFF)
#      torch.bitwise_xor(out*2e60, out, out)
#
#      return out
#
#  def cylinder_kernel_hash(coords, offsets, voxel_size):
#
#      max_theta = int(360 / voxel_size[1])
#
#      N = coords.size(0)
#      C = coords.size(1)
#      K = offsets.size(0)
#      out = torch.zeros((K, N), dtype=torch.long)
#      cur_coords = torch.zeros((K, N, 4), dtype=torch.int).cuda()
#
#      cur_coords[:] = coords
#      for k in range(K):
#          cur_coords[k, :, :3] = cur_coords[k, :, :3] + offsets[k]
#      cur_coords[:,:,:3] = (cur_coords[:,:,:3] % max_theta) & 0xFFFFFFFF
#
#      out[:,:] = 14695981039346656037 & 0xFFFFFFFFFFFFFFFF
#      for i in range(4):
#          torch.bitwise_xor(out, cur_coords[:,:,i], out)
#          out = out * (1099511628211 & 0xFFFFFFFFFFFFFFFF)
#      torch.bitwise_xor(out*2e60, out, out)
#
#      return out


def cylinder_sphash(coords: torch.Tensor,
                    offsets: Optional[torch.Tensor],
                    voxel_size 
                    ) -> torch.Tensor:
    assert coords.dtype == torch.int, coords.dtype
    assert coords.ndim == 2 and coords.shape[1] == 4, coords.shape
    coords = coords.contiguous()

    import pudb; pu.db
    max_theta = int(360//voxel_size(1))

    if offsets is None:
        if coords.device.type == 'cuda':
            return torchsparse.backend.hash_cuda(coords).cuda()
        elif coords.device.type == 'cpu':
            return torchsparse.backend.hash_cpu(coords)
        else:
            device = coords.device
            return torchsparse.backend.hash_cpu(coords.cpu()).to(device)
    else:
        assert offsets.dtype == torch.int, offsets.dtype
        assert offsets.ndim == 2 and offsets.shape[1] == 3, offsets.shape
        offsets = offsets.contiguous()

        assert coords.device.type == 'cuda', 'Cylinder mode is supported only on GPU!'
        return torchsparse.backend.cylinder_kernel_hash_cuda(coords, offsets, max_theta).cuda()
            


def sphash(coords: torch.Tensor,
           offsets: Optional[torch.Tensor] = None,
           ) -> torch.Tensor:
    assert coords.dtype == torch.int, coords.dtype
    assert coords.ndim == 2 and coords.shape[1] == 4, coords.shape
    coords = coords.contiguous()

    # TODO(Zhijian): We might be able to merge `hash_kernel` and `hash`.
    if offsets is None:
        if coords.device.type == 'cuda':
            return torchsparse.backend.hash_cuda(coords).cuda()
        elif coords.device.type == 'cpu':
            return torchsparse.backend.hash_cpu(coords)
        else:
            device = coords.device
            return torchsparse.backend.hash_cpu(coords.cpu()).to(device)
    else:
        assert offsets.dtype == torch.int, offsets.dtype
        assert offsets.ndim == 2 and offsets.shape[1] == 3, offsets.shape
        offsets = offsets.contiguous()

        if coords.device.type == 'cuda':
            return torchsparse.backend.kernel_hash_cuda(coords, offsets).cuda()
        elif coords.device.type == 'cpu':
            return torchsparse.backend.kernel_hash_cpu(coords, offsets)
        else:
            device = coords.device
            return torchsparse.backend.kernel_hash_cpu(coords.cpu(),
                                                       offsets.cpu()).to(device)
