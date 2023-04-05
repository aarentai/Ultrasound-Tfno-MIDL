from lazy_imports import np
from lazy_imports import torch
from util import diff
from data.convert import get_framework

# def riem_vec_norm(vec, g):
#   # Compute the Riemannian norm of a vector based on a metric, g
#     nrm = 1.0 / np.sqrt(np.einsum('...i,...ij,...j',vec,g,vec))
#     return (np.einsum('...i,...->...i',vec,nrm))
def get_christoffel_symbol_2d(metric_mat, mask, differential_accuracy=2):
    fw, fw_name = get_framework(metric_mat)
    if fw_name=='torch':
        tensor_mat = torch.inverse(metric_mat)
    if fw_name=='numpy':
        tensor_mat = np.linalg.inv(metric_mat)
    
    go11, go12, go21, go22 = metric_mat[...,0,0], metric_mat[...,0,1], metric_mat[...,1,0], metric_mat[...,1,1]
    gi11, gi12, gi21, gi22 = tensor_mat[...,0,0], tensor_mat[...,0,1], tensor_mat[...,1,0], tensor_mat[...,1,1]

    d1_go11 = diff.get_first_order_derivative(go11, direction=0, accuracy=differential_accuracy)
    d1_go12 = diff.get_first_order_derivative(go12, direction=0, accuracy=differential_accuracy)
    d1_go21 = diff.get_first_order_derivative(go21, direction=0, accuracy=differential_accuracy)
    d1_go22 = diff.get_first_order_derivative(go22, direction=0, accuracy=differential_accuracy)
    d2_go11 = diff.get_first_order_derivative(go11, direction=1, accuracy=differential_accuracy)
    d2_go12 = diff.get_first_order_derivative(go12, direction=1, accuracy=differential_accuracy)
    d2_go21 = diff.get_first_order_derivative(go21, direction=1, accuracy=differential_accuracy)
    d2_go22 = diff.get_first_order_derivative(go22, direction=1, accuracy=differential_accuracy)
    
#     if fw_name=='torch':
#         d1_go11, d2_go11 = diff.gradient_mask_2d(go11.numpy(), mask.numpy())
#         d1_go12, d2_go12 = diff.gradient_mask_2d(go12.numpy(), mask.numpy())
#         d1_go21, d2_go21 = diff.gradient_mask_2d(go21.numpy(), mask.numpy())
#         d1_go22, d2_go22 = diff.gradient_mask_2d(go22.numpy(), mask.numpy())
#         d1_go11, d2_go11, d1_go12, d2_go12, d1_go21, d2_go21, d1_go22, d2_go22 = torch.from_numpy(d1_go11), torch.from_numpy(d2_go11), torch.from_numpy(d1_go12), torch.from_numpy(d2_go12), torch.from_numpy(d1_go21), torch.from_numpy(d2_go21), torch.from_numpy(d1_go22), torch.from_numpy(d2_go22)
        
#     if fw_name=='numpy':
#         d1_go11, d2_go11 = diff.gradient_mask_2d(go11, mask)
#         d1_go12, d2_go12 = diff.gradient_mask_2d(go12, mask)
#         d1_go21, d2_go21 = diff.gradient_mask_2d(go21, mask)
#         d1_go22, d2_go22 = diff.gradient_mask_2d(go22, mask)
    
    gamma1 = fw.zeros_like(metric_mat)
    gamma2 = fw.zeros_like(metric_mat)
    gamma1[...,0,0] = (gi11*(d1_go11+d1_go11-d1_go11)+gi12*(d1_go12+d1_go12-d2_go11))*0.5
    gamma1[...,0,1] = (gi11*(d1_go21+d2_go11-d1_go12)+gi12*(d1_go22+d2_go12-d2_go12))*0.5
    gamma1[...,1,0] = (gi11*(d2_go11+d1_go21-d1_go21)+gi12*(d2_go12+d1_go22-d2_go21))*0.5
    gamma1[...,1,1] = (gi11*(d2_go21+d2_go21-d1_go22)+gi12*(d2_go22+d2_go22-d2_go22))*0.5
    gamma2[...,0,0] = (gi21*(d1_go11+d1_go11-d1_go11)+gi22*(d1_go12+d1_go12-d2_go11))*0.5
    gamma2[...,0,1] = (gi21*(d1_go21+d2_go11-d1_go12)+gi22*(d1_go22+d2_go12-d2_go12))*0.5
    gamma2[...,1,0] = (gi21*(d2_go11+d1_go21-d1_go21)+gi22*(d2_go12+d1_go22-d2_go21))*0.5
    gamma2[...,1,1] = (gi21*(d2_go21+d2_go21-d1_go22)+gi22*(d2_go22+d2_go22-d2_go22))*0.5
    
    return gamma1, gamma2

def get_jacobian_2d(vector_lin, mask, differential_accuracy=2):
    fw, fw_name = get_framework(vector_lin)
    v1, v2 = vector_lin[0], vector_lin[1]
    
    dv = fw.zeros((*vector_lin.shape[1:], 2, 2))
    dv[...,0,0] = diff.get_first_order_derivative(v1, direction=0, accuracy=differential_accuracy)
    dv[...,0,1] = diff.get_first_order_derivative(v1, direction=1, accuracy=differential_accuracy)
    dv[...,1,0] = diff.get_first_order_derivative(v2, direction=0, accuracy=differential_accuracy)
    dv[...,1,1] = diff.get_first_order_derivative(v2, direction=1, accuracy=differential_accuracy)

#     if fw_name=='torch':
#         dv = np.zeros((*vector_lin.shape[1:], 2, 2))
#         dv[...,0,0], dv[...,0,1] = diff.gradient_mask_2d(v1.numpy(), mask.numpy())
#         dv[...,1,0], dv[...,1,1] = diff.gradient_mask_2d(v2.numpy(), mask.numpy())
#         dv = torch.from_numpy(dv).float()
        
#     if fw_name=='numpy':
#         dv = np.zeros((*vector_lin.shape[1:], 2, 2))
#         dv[...,0,0], dv[...,0,1] = diff.gradient_mask_2d(v1, mask)
#         dv[...,1,0], dv[...,1,1] = diff.gradient_mask_2d(v2, mask)
        
    return dv

def covariant_derivative_2d(vector_lin, metric_mat, mask, differential_accuracy=2):
    """
    Calculate covariant derivative w.r.t vector_lin and metric_mat
    Args:
        vector_lin, torch.Tensor, of shape [2, h, w]
        metric_mat, torch.Tensor, of shape [h, w, 2, 2]
    Returns: 
        nabla_vv, torch.Tensor
    """
    assert vector_lin.shape[0]+1==len(vector_lin.shape), 'vector_lin should follow shape of [2, h, w]'
    assert metric_mat.shape[-1]+2==len(metric_mat.shape), 'metric_mat should follow shape of [h, w, 2, 2]'
    fw, fw_name = get_framework(vector_lin)
    
    dv = get_jacobian_2d(vector_lin, mask, differential_accuracy)
    dvv = fw.einsum('...ij,j...->i...', dv, vector_lin)
    
    vgammav = fw.zeros_like(vector_lin)
    Gamma1, Gamma2 = get_christoffel_symbol_2d(metric_mat, mask, differential_accuracy)
    vgammav[0] = fw.einsum('i...,i...->...', vector_lin, fw.einsum('...ij,j...->i...', Gamma1, vector_lin))
    vgammav[1] = fw.einsum('i...,i...->...', vector_lin, fw.einsum('...ij,j...->i...', Gamma2, vector_lin))
    
    nabla_vv = dvv + vgammav
    
    return nabla_vv

def get_christoffel_symbol_2d_batch(metric_mat, mask, differential_accuracy=2):
    fw, fw_name = get_framework(metric_mat)
    if fw_name=='torch':
        tensor_mat = torch.inverse(metric_mat)
    if fw_name=='numpy':
        tensor_mat = np.linalg.inv(metric_mat)
    
    go11, go12, go21, go22 = metric_mat[...,0,0], metric_mat[...,0,1], metric_mat[...,1,0], metric_mat[...,1,1]
    gi11, gi12, gi21, gi22 = tensor_mat[...,0,0], tensor_mat[...,0,1], tensor_mat[...,1,0], tensor_mat[...,1,1]

    d1_go11 = diff.get_first_order_derivative(go11, direction=1, accuracy=differential_accuracy)
    d1_go12 = diff.get_first_order_derivative(go12, direction=1, accuracy=differential_accuracy)
    d1_go21 = diff.get_first_order_derivative(go21, direction=1, accuracy=differential_accuracy)
    d1_go22 = diff.get_first_order_derivative(go22, direction=1, accuracy=differential_accuracy)
    d2_go11 = diff.get_first_order_derivative(go11, direction=2, accuracy=differential_accuracy)
    d2_go12 = diff.get_first_order_derivative(go12, direction=2, accuracy=differential_accuracy)
    d2_go21 = diff.get_first_order_derivative(go21, direction=2, accuracy=differential_accuracy)
    d2_go22 = diff.get_first_order_derivative(go22, direction=2, accuracy=differential_accuracy)
    
    gamma1 = fw.zeros_like(metric_mat)
    gamma2 = fw.zeros_like(metric_mat)
    gamma1[...,0,0] = (gi11*(d1_go11+d1_go11-d1_go11)+gi12*(d1_go12+d1_go12-d2_go11))*0.5
    gamma1[...,0,1] = (gi11*(d1_go21+d2_go11-d1_go12)+gi12*(d1_go22+d2_go12-d2_go12))*0.5
    gamma1[...,1,0] = (gi11*(d2_go11+d1_go21-d1_go21)+gi12*(d2_go12+d1_go22-d2_go21))*0.5
    gamma1[...,1,1] = (gi11*(d2_go21+d2_go21-d1_go22)+gi12*(d2_go22+d2_go22-d2_go22))*0.5
    gamma2[...,0,0] = (gi21*(d1_go11+d1_go11-d1_go11)+gi22*(d1_go12+d1_go12-d2_go11))*0.5
    gamma2[...,0,1] = (gi21*(d1_go21+d2_go11-d1_go12)+gi22*(d1_go22+d2_go12-d2_go12))*0.5
    gamma2[...,1,0] = (gi21*(d2_go11+d1_go21-d1_go21)+gi22*(d2_go12+d1_go22-d2_go21))*0.5
    gamma2[...,1,1] = (gi21*(d2_go21+d2_go21-d1_go22)+gi22*(d2_go22+d2_go22-d2_go22))*0.5
    
    return gamma1, gamma2

def get_jacobian_2d_batch(vector_lin, mask, differential_accuracy=2):
    fw, fw_name = get_framework(vector_lin)
    v1, v2 = vector_lin[:,0], vector_lin[:,1]
    
    dv = fw.zeros((vector_lin.shape[0],*vector_lin.shape[2:], 2, 2))
    dv[...,0,0] = diff.get_first_order_derivative(v1, direction=1, accuracy=differential_accuracy)
    dv[...,0,1] = diff.get_first_order_derivative(v1, direction=2, accuracy=differential_accuracy)
    dv[...,1,0] = diff.get_first_order_derivative(v2, direction=1, accuracy=differential_accuracy)
    dv[...,1,1] = diff.get_first_order_derivative(v2, direction=2, accuracy=differential_accuracy)
        
    return dv

def covariant_derivative_2d_batch(vector_lin, metric_mat, mask, differential_accuracy=2):
    """
    Calculate covariant derivative w.r.t vector_lin and metric_mat
    Args:
        vector_lin, torch.Tensor, of shape [b, 2, h, w]
        metric_mat, torch.Tensor, of shape [b, h, w, 2, 2]
    Returns: 
        nabla_vv, torch.Tensor
    """
    assert vector_lin.shape[1]+2==len(vector_lin.shape), 'vector_lin should follow shape of [b, 2, h, w]'
    assert metric_mat.shape[-1]+3==len(metric_mat.shape), 'metric_mat should follow shape of [b, h, w, 2, 2]'
    fw, fw_name = get_framework(vector_lin)
    
    dv = get_jacobian_2d_batch(vector_lin, mask, differential_accuracy)
    dvv = fw.einsum('b...ij,bj...->bi...', dv, vector_lin)
    
    vgammav = fw.zeros_like(vector_lin)
    Gamma1, Gamma2 = get_christoffel_symbol_2d_batch(metric_mat, mask, differential_accuracy)
    vgammav[:,0] = fw.einsum('bi...,bi...->b...', vector_lin, fw.einsum('b...ij,bj...->bi...', Gamma1, vector_lin))
    vgammav[:,1] = fw.einsum('bi...,bi...->b...', vector_lin, fw.einsum('b...ij,bj...->bi...', Gamma2, vector_lin))
    
    nabla_vv = dvv + vgammav
    
    return nabla_vv
'''torch version'''
# def get_christoffel_symbol(metric_mat, mask, differential_accuracy=2):
#     tensor_mat = torch.inverse(metric_mat)
    
#     go11, go12, go21, go22 = metric_mat[...,0,0], metric_mat[...,0,1], metric_mat[...,1,0], metric_mat[...,1,1]
#     gi11, gi12, gi21, gi22 = tensor_mat[...,0,0], tensor_mat[...,0,1], tensor_mat[...,1,0], tensor_mat[...,1,1]

# #     d1_go11 = diff.get_first_order_derivative(go11, direction=0, accuracy=differential_accuracy)
# #     d1_go12 = diff.get_first_order_derivative(go12, direction=0, accuracy=differential_accuracy)
# #     d1_go21 = diff.get_first_order_derivative(go21, direction=0, accuracy=differential_accuracy)
# #     d1_go22 = diff.get_first_order_derivative(go22, direction=0, accuracy=differential_accuracy)
# #     d2_go11 = diff.get_first_order_derivative(go11, direction=1, accuracy=differential_accuracy)
# #     d2_go12 = diff.get_first_order_derivative(go12, direction=1, accuracy=differential_accuracy)
# #     d2_go21 = diff.get_first_order_derivative(go21, direction=1, accuracy=differential_accuracy)
# #     d2_go22 = diff.get_first_order_derivative(go22, direction=1, accuracy=differential_accuracy)
    
#     d1_go11, d2_go11 = diff.gradient_mask_2d(go11, mask)
#     d1_go12, d2_go12 = diff.gradient_mask_2d(go12, mask)
#     d1_go21, d2_go21 = diff.gradient_mask_2d(go21, mask)
#     d1_go22, d2_go22 = diff.gradient_mask_2d(go22, mask)
#     d1_go11, d2_go11, d1_go12, d2_go12, d1_go21, d2_go21, d1_go22, d2_go22 = torch.from_numpy(d1_go11), torch.from_numpy(d2_go11), torch.from_numpy(d1_go12), torch.from_numpy(d2_go12), torch.from_numpy(d1_go21), torch.from_numpy(d2_go21), torch.from_numpy(d1_go22), torch.from_numpy(d2_go22)
    
#     gamma1 = torch.zeros_like(metric_mat)
#     gamma2 = torch.zeros_like(metric_mat)
#     gamma1[...,0,0] = (gi11*(d1_go11+d1_go11-d1_go11)+gi12*(d1_go12+d1_go12-d2_go11))*0.5
#     gamma1[...,0,1] = (gi11*(d1_go21+d2_go11-d1_go12)+gi12*(d1_go22+d2_go12-d2_go12))*0.5
#     gamma1[...,1,0] = (gi11*(d2_go11+d1_go21-d1_go21)+gi12*(d2_go12+d1_go22-d2_go21))*0.5
#     gamma1[...,1,1] = (gi11*(d2_go21+d2_go21-d1_go22)+gi12*(d2_go22+d2_go22-d2_go22))*0.5
#     gamma2[...,0,0] = (gi21*(d1_go11+d1_go11-d1_go11)+gi22*(d1_go12+d1_go12-d2_go11))*0.5
#     gamma2[...,0,1] = (gi21*(d1_go21+d2_go11-d1_go12)+gi22*(d1_go22+d2_go12-d2_go12))*0.5
#     gamma2[...,1,0] = (gi21*(d2_go11+d1_go21-d1_go21)+gi22*(d2_go12+d1_go22-d2_go21))*0.5
#     gamma2[...,1,1] = (gi21*(d2_go21+d2_go21-d1_go22)+gi22*(d2_go22+d2_go22-d2_go22))*0.5
    
#     return gamma1, gamma2

# def get_jacobian(vector_lin, mask, differential_accuracy=2):
#     v1, v2 = vector_lin[0], vector_lin[1]
    
# #     dv = torch.zeros((*vector_lin.shape[1:], 2, 2))
# #     dv[...,0,0] = diff.get_first_order_derivative(v1, direction=0, accuracy=differential_accuracy)
# #     dv[...,0,1] = diff.get_first_order_derivative(v1, direction=1, accuracy=differential_accuracy)
# #     dv[...,1,0] = diff.get_first_order_derivative(v2, direction=0, accuracy=differential_accuracy)
# #     dv[...,1,1] = diff.get_first_order_derivative(v2, direction=1, accuracy=differential_accuracy)
    
#     dv = np.zeros((*vector_lin.shape[1:], 2, 2))
#     dv[...,0,0], dv[...,0,1] = diff.gradient_mask_2d(v1, mask)
#     dv[...,1,0], dv[...,1,1] = diff.gradient_mask_2d(v2, mask)
#     dv = torch.from_numpy(dv).float()
    
#     return dv

# def covariant_derivative_2d(vector_lin, metric_mat, mask, differential_accuracy=2):
#     """
#     Calculate covariant derivative w.r.t vector_lin and metric_mat
#     Args:
#         vector_lin, torch.Tensor, of shape [2, h, w]
#         metric_mat, torch.Tensor, of shape [h, w, 2, 2]
#     Returns: 
#         nabla_vv, torch.Tensor
#     """
#     assert vector_lin.shape[0]+1==len(vector_lin.shape), 'vector_lin should follow shape of [2, h, w]'
#     assert metric_mat.shape[-1]+2==len(metric_mat.shape), 'metric_mat should follow shape of [h, w, 2, 2]'
    
#     dv = get_jacobian(vector_lin, mask, differential_accuracy)
#     dvv = torch.einsum('...ij,j...->i...', dv, vector_lin)
    
#     vgammav = torch.zeros_like(vector_lin)
#     Gamma1, Gamma2 = get_christoffel_symbol(metric_mat, mask, differential_accuracy)
#     vgammav[0] = torch.einsum('i...,i...->...', vector_lin, torch.einsum('...ij,j...->i...', Gamma1, vector_lin))
#     vgammav[1] = torch.einsum('i...,i...->...', vector_lin, torch.einsum('...ij,j...->i...', Gamma2, vector_lin))
    
#     nabla_vv = dvv + vgammav
    
#     return nabla_vv

'''old'''
# def covariant_derivative_2d_old(vector_lin, metric_mat, differential_accuracy=2):
#     """
#     Calculate covariant derivative w.r.t vector_lin and metric_mat
#     Args:
#         vector_lin, torch.Tensor, of shape [2, h, w]
#         metric_mat, torch.Tensor, of shape [h, w, 2, 2]
#     Returns: 
#         nabla_vv, torch.Tensor
#     """
#     assert vector_lin.shape[0]+1==len(vector_lin.shape), 'vector_lin should follow shape of [2, h, w]'
#     assert metric_mat.shape[-1]+2==len(metric_mat.shape), 'metric_mat should follow shape of [h, w, 2, 2]'
    
#     tensor_mat = torch.inverse(metric_mat)
    
#     go11, go12, go21, go22 = metric_mat[...,0,0], metric_mat[...,0,1], metric_mat[...,1,0], metric_mat[...,1,1]
#     gi11, gi12, gi21, gi22 = tensor_mat[...,0,0], tensor_mat[...,0,1], tensor_mat[...,1,0], tensor_mat[...,1,1]

# #     d1_go11 = (torch.roll(go11,-1, 0)-torch.roll(go11,1, 0))/2
# #     d1_go12 = (torch.roll(go12,-1, 0)-torch.roll(go12,1, 0))/2
# #     d1_go21 = (torch.roll(go21,-1, 0)-torch.roll(go21,1, 0))/2
# #     d1_go22 = (torch.roll(go22,-1, 0)-torch.roll(go22,1, 0))/2
# #     d2_go11 = (torch.roll(go11,-1, 1)-torch.roll(go11,1, 1))/2
# #     d2_go12 = (torch.roll(go12,-1, 1)-torch.roll(go12,1, 1))/2
# #     d2_go21 = (torch.roll(go21,-1, 1)-torch.roll(go21,1, 1))/2
# #     d2_go22 = (torch.roll(go22,-1, 1)-torch.roll(go22,1, 1))/2
#     d1_go11 = diff.get_first_order_derivative(go11, direction=0, accuracy=differential_accuracy)
#     d1_go12 = diff.get_first_order_derivative(go12, direction=0, accuracy=differential_accuracy)
#     d1_go21 = diff.get_first_order_derivative(go21, direction=0, accuracy=differential_accuracy)
#     d1_go22 = diff.get_first_order_derivative(go22, direction=0, accuracy=differential_accuracy)
#     d2_go11 = diff.get_first_order_derivative(go11, direction=1, accuracy=differential_accuracy)
#     d2_go12 = diff.get_first_order_derivative(go12, direction=1, accuracy=differential_accuracy)
#     d2_go21 = diff.get_first_order_derivative(go21, direction=1, accuracy=differential_accuracy)
#     d2_go22 = diff.get_first_order_derivative(go22, direction=1, accuracy=differential_accuracy)

#     gamma1_11 = (gi11*(d1_go11+d1_go11-d1_go11)+gi12*(d1_go12+d1_go12-d2_go11))*0.5
#     gamma1_12 = (gi11*(d1_go21+d2_go11-d1_go12)+gi12*(d1_go22+d2_go12-d2_go12))*0.5
#     gamma1_21 = (gi11*(d2_go11+d1_go21-d1_go21)+gi12*(d2_go12+d1_go22-d2_go21))*0.5
#     gamma1_22 = (gi11*(d2_go21+d2_go21-d1_go22)+gi12*(d2_go22+d2_go22-d2_go22))*0.5
#     gamma2_11 = (gi21*(d1_go11+d1_go11-d1_go11)+gi22*(d1_go12+d1_go12-d2_go11))*0.5
#     gamma2_12 = (gi21*(d1_go21+d2_go11-d1_go12)+gi22*(d1_go22+d2_go12-d2_go12))*0.5
#     gamma2_21 = (gi21*(d2_go11+d1_go21-d1_go21)+gi22*(d2_go12+d1_go22-d2_go21))*0.5
#     gamma2_22 = (gi21*(d2_go21+d2_go21-d1_go22)+gi22*(d2_go22+d2_go22-d2_go22))*0.5

#     v1, v2 = vector_lin[0], vector_lin[1]
# #     d1_v1 = (torch.roll(v1, -1, 0)-torch.roll(v1, 1, 0))/2
# #     d1_v2 = (torch.roll(v2, -1, 0)-torch.roll(v2, 1, 0))/2
# #     d2_v1 = (torch.roll(v1, -1, 1)-torch.roll(v1, 1, 1))/2
# #     d2_v2 = (torch.roll(v2, -1, 1)-torch.roll(v2, 1, 1))/2
#     d1_v1 = diff.get_first_order_derivative(v1, direction=0, accuracy=differential_accuracy)
#     d1_v2 = diff.get_first_order_derivative(v2, direction=0, accuracy=differential_accuracy)
#     d2_v1 = diff.get_first_order_derivative(v1, direction=1, accuracy=differential_accuracy)
#     d2_v2 = diff.get_first_order_derivative(v2, direction=1, accuracy=differential_accuracy)

#     nabla_vv = torch.zeros_like(vector_lin)    
#     nabla_vv[0] = v1*d1_v1+v2*d2_v1+gamma1_11*v1*v1+gamma1_12*v1*v2+gamma1_21*v2*v1+gamma1_22*v2*v2
#     nabla_vv[1] = v1*d1_v2+v2*d2_v2+gamma2_11*v1*v1+gamma2_12*v1*v2+gamma2_21*v2*v1+gamma2_22*v2*v2
#     dvv = torch.zeros_like(vector_lin)   
#     dvv[0] = v1*d1_v1+v2*d2_v1
#     dvv[1] = v1*d1_v2+v2*d2_v2
#     vgammav = torch.zeros_like(vector_lin)   
#     vgammav[0] = gamma1_11*v1*v1+gamma1_12*v1*v2+gamma1_21*v2*v1+gamma1_22*v2*v2
#     vgammav[1] = gamma2_11*v1*v1+gamma2_12*v1*v2+gamma2_21*v2*v1+gamma2_22*v2*v2
#     return nabla_vv, dvv, vgammav