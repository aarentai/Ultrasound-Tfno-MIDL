import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import pdb, time
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
from skimage import data
from scipy.ndimage import zoom
from scipy import ndimage, datasets

def A_laplace(x, h=1, bc='dirichlet'):
    dim = len(x.shape)
    if dim not in [2, 3]:
        raise Exception("Dimension of the field should be either 2 or 3.")
#     don't forget to new the matrix as complex_ type, or it will raise "ComplexWarning: Casting complex values to real discards the imaginary part"
    x_ext = np.zeros([c+2 for c in x.shape], dtype = 'complex_')
    
    if dim==2:
        x_ext[1:-1,1:-1] = x
        if bc=='neumann':
            x_ext[-1,1:-1] = x[-1,:]
            x_ext[0,1:-1] = x[0,:]
            x_ext[1:-1,-1] = x[:,-1]
            x_ext[1:-1,0] = x[:,0]
        if bc=='periodic':
            x_ext[-1,1:-1] = x[0,:]
            x_ext[0,1:-1] = x[-1,:]
            x_ext[1:-1,-1] = x[:,0]
            x_ext[1:-1,0] = x[:,-1]
        Lx_ext = (np.roll(x_ext, -1, axis=0)+np.roll(x_ext, 1, axis=0)+np.roll(x_ext, -1, axis=1)+np.roll(x_ext, 1, axis=1)-4*x_ext)/h**2
        Lx = Lx_ext[1:-1,1:-1]
        
    if dim==3:
        x_ext[1:-1,1:-1,1:-1] = x
        if bc=='neumann':
            x_ext[-1,1:-1,1:-1] = x[-1,:,:]
            x_ext[0,1:-1,1:-1] = x[0,:,:]
            x_ext[1:-1,-1,1:-1] = x[:,-1,:]
            x_ext[1:-1,0,1:-1] = x[:,0,:]
            x_ext[1:-1,1:-1,-1] = x[:,:,-1]
            x_ext[1:-1,1:-1,0] = x[:,:,0]
        if bc=='periodic':
            x_ext[-1,1:-1,1:-1] = x[0,:,:]
            x_ext[0,1:-1,1:-1] = x[-1,:,:]
            x_ext[1:-1,-1,1:-1] = x[:,0,:]
            x_ext[1:-1,0,1:-1] = x[:,-1,:]
            x_ext[1:-1,1:-1,-1] = x[:,:,0]
            x_ext[1:-1,1:-1,0] = x[:,:,-1]            
        Lx_ext = (np.roll(x_ext, -1, axis=0)+np.roll(x_ext, 1, axis=0)+np.roll(x_ext, -1, axis=1)+np.roll(x_ext, 1, axis=1)+np.roll(x_ext, -1, axis=2)+np.roll(x_ext, 1, axis=2)-6*x_ext)/h**2
        Lx = Lx_ext[1:-1,1:-1,1:-1]
        
    return Lx

def AA_laplace(x, h=1, bc='dirichlet'):
    Ax = A_laplace(x, h, bc)
    AAx = A_laplace(Ax, h, bc)
    return AAx

'''Refering to matlab pcg'''
def laplace_cgsolver_psd(Ab, h, iteration, bc='dirichlet'):
    residual_norm_list = []
    x = np.zeros_like(Ab)
    r = Ab - AA_laplace(x, h, bc)
    rho = 1
    for i in range(iteration):
        rho1 = rho
        rho = np.sum(r**2)
        if i==0: 
            p = np.copy(r)
        else:
            beta = rho/rho1
            p = r+beta*p
        Ap = AA_laplace(p, h, bc)
        alpha = rho/np.sum(p*Ap)
        x = x+alpha*p
        r = r-alpha*Ap
        residual_norm_list.append(np.linalg.norm(r))
    return x, residual_norm_list

def A_helmholtz(x, k, h=1, bc='dirichlet'):
    Lx = A_laplace(x, h, bc)
    Ax = -(Lx+np.multiply(k, x))
    return Ax

def AA_helmholtz(x, k, h=1, bc='dirichlet'):
    Ax = A_helmholtz(x, k, h, bc)
    AAx = A_helmholtz(Ax, k, h, bc)
    return AAx

'''Refering to matlab pcg'''
def helmholtz_cgsolver_psd(Ab, k, h, iteration, bc='dirichlet'):
    residual_norm_list = []
    x = np.zeros_like(Ab)
    r = Ab - AA_helmholtz(x, k, h, bc)
    rho = 1
    for i in range(iteration):
        rho1 = rho
        rho = np.sum(r**2)
        if i==0: 
            p = np.copy(r)
        else:
            beta = rho/rho1
            p = r+beta*p
        Ap = AA_helmholtz(p, k, h, bc)
        alpha = rho/np.sum(p*Ap)
        x = x+alpha*p
        r = r-alpha*Ap
        residual_norm_list.append(np.linalg.norm(r))
    return x, residual_norm_list

def helmholtz_ftsolver(b, k, h=1):
    '''fourier transform assuming periodic boundary condition'''
    v, w = np.meshgrid(np.linspace(0, 1, b.shape[0], endpoint=False), 
                       np.linspace(0, 1, b.shape[1], endpoint=False), 
                       indexing='ij')
    L = (2*np.cos(2*np.pi*v)+2*np.cos(2*np.pi*w)-4)/h**2+k
    B = np.fft.fftn(b)
    x = np.fft.ifftn(np.multiply(B, 1./L)).real
    return x

def A_westervelt(x, k, a, h=1, bc='dirichlet'):
    Lx = A_laplace(x, h, bc)
#     remember k and delta are the coefficient of x, not constant
    Ax = -(Lx+np.multiply(k+a*1j,x))
    return Ax

def AA_westervelt(x, k, a, h=1, bc='dirichlet'):
    Ax = A_westervelt(x, k, a, h, bc)
#     A*Ax=A*b, where A_westervelt(Ax, k, -delta, h, bc) is the A* (conjugate)
    AAx = A_westervelt(Ax, k, -a, h, bc)
    return AAx

'''Refering to matlab pcg'''
def westervelt_cgsolver_psd(Ab, k, a, h, iteration, bc='dirichlet'):
    residual_norm_list = []
    x = np.zeros_like(Ab)
    r = Ab - AA_westervelt(x, k, a, h, bc)
    rho = 1
    for i in range(iteration):
        rho1 = rho
        rho = np.sum(r**2)
        if i==0: 
            p = np.copy(r)
        else:
            beta = rho/rho1
            p = r+beta*p
        Ap = AA_westervelt(p, k, a, h, bc)
        alpha = rho/np.sum(p*Ap)
        x = x+alpha*p
        r = r-alpha*Ap
        residual_norm_list.append(np.linalg.norm(r))
    return x, residual_norm_list

'''my version'''
def helmholtz_cgsolver_psd_fd(Ab, k, h, iteration, bc='dirichlet'):
    '''initialize pressure'''
    x = np.zeros_like(Ab)
#     b = A_helmholtz(b, k, h, bc)
    '''get residual'''
    r = Ab - AA_helmholtz(x, k, h, bc)
    '''initial search direction'''
    p = np.copy(r)
    for k in range(iteration):
        alpha = np.sum(r**2)/np.sum(p*AA_helmholtz(p, k, h, bc))
        x = x + alpha*p
        r_new = r - alpha*AA_helmholtz(p, k, h, bc)
        beta = np.sum(r_new**2)/np.sum(r**2)
        p = r_new + beta*p
        r = r_new
    return x

def fftIndgen(n):
    a = list(range(0, int(n/2+1)))
    b = list(range(1, int(n/2)))
    b = b[::-1]
    b = [-i for i in b]
    return a + b

def gaussian_random_field(Pk = lambda k : k**-3.0, size = 100):
    def Pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(Pk(np.sqrt(kx**2 + ky**2)))
    noise = np.fft.fft2(np.random.normal(size = (size, size)))
    amplitude = np.zeros((size,size))
    for i, kx in enumerate(fftIndgen(size)):
        for j, ky in enumerate(fftIndgen(size)):            
            amplitude[i, j] = Pk2(kx, ky)
    return np.fft.ifft2(noise * amplitude)


# def gradient(u, h=1, bc='dirichlet'):
#     dim = len(u.shape)
#     if dim not in [2, 3]:
#         raise Exception("Dimension of the field should be either 2 or 3.")
# #     don't forget to new the matrix as complex_ type, or it will raise "ComplexWarning: Casting complex values to real discards the imaginary part"
#     u_ext = np.zeros([c+2 for c in u.shape], dtype = 'complex_')
#     Lu_ext = np.zeros((*u_ext.shape, dim))
    
#     if dim==2:
#         u_ext[1:-1,1:-1] = u
#         if bc=='neumann':
#             u_ext[-1,1:-1] = u[-1,:]
#             u_ext[0,1:-1] = u[0,:]
#             u_ext[1:-1,-1] = u[:,-1]
#             u_ext[1:-1,0] = u[:,0]
#         if bc=='periodic':
#             u_ext[-1,1:-1] = u[0,:]
#             u_ext[0,1:-1] = u[-1,:]
#             u_ext[1:-1,-1] = u[:,0]
#             u_ext[1:-1,0] = u[:,-1]
#         Lu_ext[...,0] = (np.roll(u_ext, -1, axis=0)-np.roll(u_ext, 1, axis=0))/h
#         Lu_ext[...,1] = (np.roll(u_ext, -1, axis=1)-np.roll(u_ext, 1, axis=1))/h
#         Lu = Lu_ext[1:-1,1:-1]
        
#     if dim==3:
#         u_ext[1:-1,1:-1,1:-1] = u
#         if bc=='neumann':
#             u_ext[-1,1:-1,1:-1] = u[-1,:,:]
#             u_ext[0,1:-1,1:-1] = u[0,:,:]
#             u_ext[1:-1,-1,1:-1] = u[:,-1,:]
#             u_ext[1:-1,0,1:-1] = u[:,0,:]
#             u_ext[1:-1,1:-1,-1] = u[:,:,-1]
#             u_ext[1:-1,1:-1,0] = u[:,:,0] 
#         if bc=='periodic':
#             u_ext[-1,1:-1,1:-1] = u[0,:,:]
#             u_ext[0,1:-1,1:-1] = u[-1,:,:]
#             u_ext[1:-1,-1,1:-1] = u[:,0,:]
#             u_ext[1:-1,0,1:-1] = u[:,-1,:]
#             u_ext[1:-1,1:-1,-1] = u[:,:,0]
#             u_ext[1:-1,1:-1,0] = u[:,:,-1]
#         Lu_ext[...,0] = (np.roll(u_ext, -1, axis=0)-np.roll(u_ext, 1, axis=0))/h
#         Lu_ext[...,1] = (np.roll(u_ext, -1, axis=1)-np.roll(u_ext, 1, axis=1))/h
#         Lu_ext[...,2] = (np.roll(u_ext, -1, axis=2)-np.roll(u_ext, 1, axis=2))/h
#         Lu = Lu_ext[1:-1,1:-1,1:-1]
        
#     return Lu
