#set things up
import numpy  as np                       
import time, sys  
import matplotlib.pyplot as pl
from matplotlib import cm
from scipy.interpolate import make_interp_spline, BSpline
import pdb
 
def main():
    Ra        = 10^5    # set the Rayleigh number, viscosity will be a free parameter to match it
    Tref      = 273     # reference Temperature (K)
    Tleft     = 352     # left (hot) dirichlet boundary condition on temperature
    Tright    = 283     # right (cold) dirichlet boundary condition on temperature
    beta      = 0.00369  # thermal expansion coefficient (1/K)  0.00369 for air at 0 deg C, 1 Bar 
    gmag      = 9.8     # gravitational acceleration magnitude
    gx        = 0*gmag  # gravitational acceleration in x direction
    gy        = -1*gmag # gravitational acceleration in y direction
    Cp        = 1000    # specific heat capacity (J/kg/K), 1000 for air at stp
    k         = 0.026   # thermal conductivity (W/m/K), 0.026 for air
    rhoref    = 1.225   # reference density (kg/m^3), 1.225 for air at stp
    mu        = gmag*beta*k/Cp/rhoref*(Tleft-Tright)/Ra     # diffusion coefficient, really nu [m^2/s], assumes x = 1
    
    deltat    = 0.02    # time  step (stability limit checked later)
    timesteps = 30      # number of timesteps
    solver    = "ilu"
    plot      = 1
    ncv       = 3  
    ubc_south = 0.0  # velocity of south wall
    ubc_north = 0.0  # velocity of north wall
    vbc_west  = 0.0  # velocity of west  wall
    vbc_east  = 0.0  # velocity of east  wall
    
    solve(ncv,ubc_south,ubc_north,vbc_west,vbc_east,solver,plot,mu,Tref,Tleft,Tright,beta,gx,gy,Cp,k,rhoref,deltat,timesteps)
    
    return

# solve a linear system using gauss-seidel
def solve_gs(ncv_x,ncv_y,dx,dy,phi,a_east,a_west,a_north,a_south,a_p,b,n_iter,tolerance):

    res = np.zeros((ncv_y,ncv_x)) #initialize

    for n in range(n_iter):
        # update solution - internal cells
        for jcv in range(ncv_y):
            jcv_north = min(jcv+1,ncv_y-1)  # assumes a_north[ncv_y-1,icv]=0
            jcv_south = max(jcv-1,0)        # assumes a_south[0,icv]=0
            for icv in range(ncv_x):
                icv_east = min(icv+1,ncv_x-1)   # assumes a_east[jcv,ncv_x-1]=0
                icv_west = max(icv-1,0)         # assumes a_west[jcv,0]=0
                phi[jcv,icv]=(b[jcv,icv]
                              -a_east [jcv,icv]*phi[jcv,icv_east ]-a_west [jcv,icv]*phi[jcv,icv_west ] 
                          -a_north[jcv,icv]*phi[jcv_north,icv]-a_south[jcv,icv]*phi[jcv_south,icv])/a_p[jcv,icv]

        for jcv in range(ncv_y):
            jcv_north = min(jcv+1,ncv_y-1)  # assumes a_north[ncv_y-1,icv]=0
            icv_south = max(jcv-1,0)        # assumes a_south[0,icv]=0
            for icv in range(ncv_x):
                icv_east = min(icv+1,ncv_x-1)   # assumes a_east[jcv,ncv_x-1]=0
                icv_west = max(icv-1,0)         # assumes a_west[jcv,0]=0
                res[jcv,icv] = (b[jcv,icv]
                                -a_east [jcv,icv]*phi[jcv,icv_east ]-a_west [jcv,icv]*phi[jcv,icv_west ]
                        -a_north[jcv,icv]*phi[jcv_north,icv]-a_south[jcv,icv]*phi[jcv_south,icv]
                        -a_p[jcv,icv]*phi[jcv,icv])
                residual = np.sqrt(dx*dy*np.sum(res[0:ncv_y,0:ncv_x]**2))                     
        if (n == 0):
            residual0 = residual
        if residual/residual0 < tolerance:
            break
    print ("Completed Gauss-Seidel Iteration ",n,"Residual ",residual/residual0)

    return

# solve a linear system using ILU
def solve_ilu(ncv_x,ncv_y,dx,dy,phi,a_east,a_west,a_north,a_south,a_p,b,n_iter,tolerance):
    small = 1.0E-8
    alpha = 0.92 #under-relation factor...
    l_west  = np.zeros((ncv_y,ncv_x))
    l_south = np.zeros((ncv_y,ncv_x))
    l_p     = np.zeros((ncv_y,ncv_x))
    u_north = np.zeros((ncv_y,ncv_x))
    u_east  = np.zeros((ncv_y,ncv_x))
    res     = np.zeros((ncv_y,ncv_x))             

    # compute L/U coefficients

    for jcv in range(ncv_y):
        jcv_south = max(jcv-1,0) 
        for icv in range(ncv_x):
            icv_west = max(icv-1,0)         # assumes u_north[jcv,0]=0, a_west[jcv,0]=0 
            l_west [jcv,icv] = a_west [jcv,icv]/(1.+alpha*u_north[jcv,icv_west])
            l_south[jcv,icv] = a_south[jcv,icv]/(1.+alpha*u_east [jcv_south,icv] )
            p1 = alpha*l_west [jcv,icv]*u_north[jcv,icv_west]
            p2 = alpha*l_south[jcv,icv]*u_east [jcv_south,icv]            
            l_p [jcv,icv]    = (a_p[jcv,icv]+p1+p2 -l_west [jcv,icv]*u_east [jcv,icv_west]
                                -l_south[jcv,icv]*u_north[jcv_south,icv] )
            u_north[jcv,icv] = (a_north[jcv,icv]-p1)/(l_p[jcv,icv]+small)
            u_east [jcv,icv] = (a_east [jcv,icv]-p2)/(l_p[jcv,icv]+small)

    for n in range(n_iter+1):

        # compute residual
        residual = 0.0
        for jcv in range(ncv_y):
            jcv_north = min(jcv+1,ncv_y-1)  # assumes a_north[ncv_y-1,icv]=0
            jcv_south = max(jcv-1,0)        # assumes a_south[0,icv]=0
            for icv in range(ncv_x):
                icv_east = min(icv+1,ncv_x-1)   # assumes a_east[jcv,ncv_x-1]=0
                icv_west = max(icv-1,0)         # assumes a_west[jcv,0]=0
                res[jcv,icv]=(b[jcv,icv]
                              -a_east [jcv,icv]*phi[jcv,icv_east ]-a_west [jcv,icv]*phi[jcv,icv_west ]
                            -a_north[jcv,icv]*phi[jcv_north,icv]-a_south[jcv,icv]*phi[jcv_south,icv]
                            -a_p[jcv,icv]*phi[jcv,icv])
                residual=residual+res[jcv,icv]*res[jcv,icv]

                res[jcv,icv]=(res[jcv,icv]-l_south[jcv,icv]*res[jcv_south,icv]
                              -l_west [jcv,icv]*res[jcv,icv_west] )/(l_p[jcv,icv]+small)  

        residual = np.sqrt(dx*dy*residual)
        if (n == 0):
            residual0 = residual

        # back-substitution
        for jcv in range(ncv_y-1,-1,-1):
            jcv_north = min(jcv+1,ncv_y-1)  # assumes a_north[ncv_y-1,icv]=0
            for icv in range(ncv_x-1,-1,-1):
                icv_east = min(icv+1,ncv_x-1)   # assumes a_east[jcv,ncv_x-1]=0
                res[jcv,icv]=res[jcv,icv]-u_north[jcv,icv]*res[jcv_north,icv]-u_east[jcv,icv]*res[jcv,icv_east]
                phi[jcv,icv]=phi[jcv,icv]+res[jcv,icv]

        #print "It",n,"residual",residual        
        if residual/residual0 < tolerance:
            break
    #if (n==n_iter):
    #    print "ILU achieved max iteration ",n,"Residual ",residual/residual0
    
    return

# solve the Poisson equation for pressure  
def solve_p(ncv_x,ncv_y,nno_x,nno_y,h,dt,p,a_east,a_west,a_north,a_south,a_p,b,ut,vt,solver="ilu"):  

    #solve for the pressure 
    for jcv in range(ncv_y):
        jno_north =  jcv+1    
        jno_south =  jcv  
        for icv in range(ncv_x):        
            ino_east  =  icv+1 
            ino_west  =  icv 
            p[jcv,icv] = 0.0
            b[jcv,icv] = (h/dt)*( ut[jcv,ino_east]-ut[jcv,ino_west]+
                                  vt[jno_north,icv]-vt[jno_south,icv] )
     
    if (solver=="ilu"):
        solve_ilu(ncv_x,ncv_y,h,h,p,a_east,a_west,a_north,a_south,a_p,b,25,1.0E-5)
    else:
        solve_gs(ncv_x,ncv_y,h,h,p,a_east,a_west,a_north,a_south,a_p,b,25,1.0E-5)

    return    

def solve_uv(ncv_x,ncv_y,nno_x,nno_y,h,dt,ubc_south,ubc_north,vbc_west,vbc_east,u,v,ut,vt,mu,p,beta,Tref,gx,gy,T):
    
    # update the temporary x-velocity field (only internal u-cells)    
    for jcv in range(ncv_y): 
        for ino in range(1,nno_x):
            icv = ino
            jno = jcv
            # compute convective fluxes
            # compute x-velocity at the faces of the u-control volume &
            # compute y-velocity at the y-faces of the u-control volume 
            if (jcv==0): # south boundary of the domain
                if ino == 1:  # south east corner 
                    uf_north = 0.0 
                    uf_south = ubc_south
                    uf_east  = 0.5 * (u[jcv,ino] + u[jcv,ino+1])
                    uf_west  = 0.0 
                    vf_south = vbc_west 
                    vf_north = vbc_west  
                elif ino == nno_x-1:  # south west corner 
                    uf_north = 0.0 
                    uf_south = ubc_south
                    uf_east  = 0.0 
                    uf_west  = 0.5 * (u[jcv,ino] + u[jcv,ino-1])
                    vf_south = vbc_east 
                    vf_north = vbc_east
                else: #remainder of south wall 
                    uf_north = 0.5 * (u[jno+1,icv] + u[jno+1,icv-1]) 
                    uf_south = ubc_south
                    uf_east  = 0.5 * (u[jcv,ino] + u[jcv,ino+1])
                    uf_west  = 0.5 * (u[jcv,ino] + u[jcv,ino-1])
                    vf_south = 0.0  
                    vf_north = 0.5 * (v[jno+1,icv] + v[jno+1,icv-1])
                       
            elif (jcv==ncv_y-1): # north boundary of the domain
                if ino == 1:  # north east corner 
                    uf_north = ubc_north
                    uf_south = 0.0 
                    uf_east  = 0.5 * (u[jcv,ino] + u[jcv,ino+1])
                    uf_west  = 0.0 
                    vf_south = vbc_west 
                    vf_north = vbc_west 
                    
                elif ino == nno_x-1:  # north west corner 
                    uf_north = ubc_north
                    uf_south = 0.0 
                    uf_east  = 0.0 
                    uf_west  = 0.5 * (u[jcv,ino] + u[jcv,ino-1])
                    vf_south = vbc_east 
                    vf_north = vbc_east
                     
                else: # remainder of north wall 
                    uf_north = ubc_north
                    uf_south = 0.5 * (v[jno-1,icv] + v[jno-1,icv-1]) 
                    uf_east  = 0.5 * (u[jcv,ino] + u[jcv,ino+1])
                    uf_west  = 0.5 * (u[jcv,ino] + u[jcv,ino-1])
                    vf_south = 0.5 * (v[jno-1,icv] + v[jno-1,icv-1])        
                    vf_north = 0.0   
                
            elif (ino == 1) and (jcv > 0) :  # west border besides corners 
                    uf_north = 0.0 
                    uf_south = 0.0 
                    uf_east  = 0.5 * (u[jcv,ino] + u[jcv,ino+1])
                    uf_west  = 0.0 
                    vf_south = vbc_west 
                    vf_north = vbc_west 
                    
            elif (ino == nno_x-1) and (jcv > 0) :  # east border besides corners 
                    uf_north = 0.0 
                    uf_south = 0.0 
                    uf_east  = 0.0 
                    uf_west  = 0.5 * (u[jcv,ino] + u[jcv,ino-1])
                    vf_south = vbc_east 
                    vf_north = vbc_east  
                    
            else: # remainder of computational domain
                uf_north = 0.5 * (u[jcv,ino]   + u[jcv+1,ino])              
                uf_south = 0.5 * (u[jcv,ino]   + u[jcv-1,ino])
                uf_east  = 0.5 * (u[jcv,ino] + u[jcv,ino+1])
                uf_west  = 0.5 * (u[jcv,ino] + u[jcv,ino-1])  
                vf_south = 0.5 * (v[jno,icv]   + v[jno,icv-1])
                vf_north = 0.5 * (v[jno+1,icv] + v[jno+1,icv-1])
              
            # Convective Part of H        
            F_c = (uf_east**2-uf_west**2+uf_north*vf_north-uf_south*vf_south)/h
                
            # compute diffusive fluxes
            if (jcv==0):  # south boundary of the domain
                if ino == 1: # south west corner 
                    ucv_east  = u[jcv,ino+1]
                    ucv_west  = 0.0                
                    ucv_south = 2.0 * ubc_south - u[jcv,ino] # half at corner # extrapolating with u @ south boundary = 0
                    ucv_north = u[jcv+1,ino]
                    
                elif ino == nno_x-1: # south east corner 
                    ucv_east  = 0.0
                    ucv_west  = u[jcv,ino-1]                
                    ucv_south = 2.0 * ubc_south - u[jcv,ino] # half at corner # extrapolating with u @ south boundary = 0
                    ucv_north = u[jcv+1,ino]                    
                
                else: # remainder or south boundary 
                    ucv_east  = u[jcv,ino+1]
                    ucv_west  = u[jcv,ino-1]                
                    ucv_south = 2.0 * ubc_south - u[jcv,ino] # extrapolating with u @ south boundary = 0
                    ucv_north = u[jcv+1,ino] 
                    
            elif (jcv==ncv_y-1):                 
                if ino == 1: # north west corner 
                    ucv_east  = u[jcv,ino+1]
                    ucv_west  = 0.0                
                    ucv_south = u[jcv-1,ino] 
                    ucv_north = 2.0 * ubc_north - u[jcv,ino]# half at corner # extrapolating with u @ south boundary = 0
                    
                elif ino == nno_x-1: # north east corner 
                    ucv_east  = 0.0
                    ucv_west  = u[jcv,ino-1]                
                    ucv_south = u[jcv-1,ino]  
                    ucv_north = 2.0 * ubc_north - u[jcv,ino] # half at corner # extrapolating with u @ south boundary = 0
                
                else: # remainder or north boundary 
                    ucv_east  = u[jcv,ino+1]
                    ucv_west  = u[jcv,ino-1]                
                    ucv_south = u[jcv-1,ino]    # extrapolating with u @ south boundary = 0
                    ucv_north = 2.0 * ubc_north - u[jcv,ino] 
             
            elif (ino == 1) and (jcv > 0): # west wall besides cordners 
                ucv_east  = u[jcv,ino+1]  
                ucv_west  = 0.0              
                ucv_south = u[jcv-1,ino]  
                ucv_north = u[jcv+1,ino]  
            elif (ino == nno_x-1) and (jcv > 0) :  # east border besides corners   
                ucv_east  = 0.0
                ucv_west  = u[jcv,ino-1]
                ucv_south = u[jcv-1,ino]
                ucv_north = u[jcv+1,ino]
                
            else: #remainder of domain 
                ucv_north = u[jcv+1,ino]                
                ucv_south = u[jcv-1,ino]
                ucv_east = u[jcv,ino+1]
                ucv_west = u[jcv,ino-1]     
                
            # Diffusive part of H - second order central difference       
            F_d = (mu/(h**2))*(ucv_east+ucv_west+ucv_north+ucv_south-4.0*u[jcv,ino])
            
            # Compute temperature forcing function 
            F_t = -beta*(T[jcv,icv]-Tref)*gx
            
            # update the temporary velocity
            ut[jcv,ino] = u[jcv,ino] + dt * (-F_c+F_d+F_t) 
                
    # update the temporary y-velocity field (only internal v-cells)    
    for jno in range (1,nno_y): 
        for icv in range(ncv_x):
            ino = icv
            jcv = jno
            # compute convective fluxes 
            # compute y-velocity at the faces of the v-control volume &
            # compute x-velocity at the x-faces of the v-control volume 
            if (jcv==1): # south boundary of the domain
                if ino == 0: # south east corner 
                    vf_north  = 0.5 * (v[jno,icv] + v[jno+1,icv])
                    vf_south  = 0.0                   
                    vf_east   = 0.5 * (v[jno,icv]+v[jno,icv+1])
                    vf_west   = vbc_west
                    uf_west   = ubc_south
                    uf_east   = ubc_south                    
                elif ino == ncv_x-1: # south west corner 
                    vf_north  = 0.5 * (v[jno,icv] + v[jno+1,icv])
                    vf_south  = 0.0                    
                    vf_east   = vbc_east 
                    vf_west   = 0.5 * (v[jno,icv]+v[jno,icv-1]) 
                    uf_west   = ubc_south
                    uf_east   = ubc_south  
                else: # remainder of south wall 
                    vf_north  = 0.5 * (v[jno,icv] + v[jno+1,icv])
                    vf_south  = 0.0                     
                    vf_east   = 0.5 * (v[jno,icv]+v[jno,icv+1])
                    vf_west   = 0.5 * (v[jno,icv]+v[jno,icv-1])  
                    uf_west   = ubc_south
                    uf_east   = ubc_south
                    
            elif (jcv==nno_y-1): # north boundary of the domain
                if ino == 0: # north east corner 
                    vf_north  = 0.0
                    vf_south  = 0.5 * (v[jno,icv] + v[jno-1,icv])                 
                    vf_east   = 0.5 * (v[jno,icv]+v[jno,icv+1])
                    vf_west   = vbc_west
                    uf_west   = ubc_north
                    uf_east   = ubc_north                   
                if ino == ncv_x-1: # north west corner 
                    vf_north  = 0.0   
                    vf_south  = 0.5 * (v[jno,icv] + v[jno-1,icv])                  
                    vf_east   = vbc_east 
                    vf_west   = 0.5 * (v[jno,icv]+v[jno,icv-1]) 
                    uf_west   = ubc_north
                    uf_east   = ubc_north  
                else: # remainder of north wall 
                    vf_north  = 0.0
                    vf_south  = 0.5 * (v[jno,icv] + v[jno-1,icv])                   
                    vf_east   = 0.5 * (v[jno,icv]+v[jno,icv+1])
                    vf_west   = 0.5 * (v[jno,icv]+v[jno,icv-1])  
                    uf_west   = ubc_north
                    uf_east   = ubc_north                 
                 
            elif (icv == 0) and (jcv > 0) :  # west border besides corners 
                vf_north = 0.5 * (v[jno,icv] + v[jno+1,icv+1]) 
                vf_south = 0.5 * (v[jno,icv] + v[jno+1,icv-1]) 
                vf_east  = 0.5 * (v[jno,icv] + v[jno,icv+1])
                vf_west  = vbc_west     
                uf_east  = 0.5 * (u[jno,icv] + u[jno,icv+1])   
                uf_west  = 0.0  
        
            elif (icv == ncv_x-1) and (jcv > 0) :  # east border besides corners 
                vf_north = 0.5 * (v[jno,icv] + v[jno+1,icv+1])
                vf_south = 0.5 * (v[jno,icv] + v[jno+1,icv-1])
                vf_east  = vbc_east 
                vf_west  = 0.5 * (v[jno,icv] + v[jno,icv-1])    
                uf_east  = 0.0 
                uf_west  = 0.5 * (u[jno,icv] + u[jno,icv-1])
            
            else: 
                vf_north  = 0.5 * (v[jno,icv] + v[jno+1,icv])
                vf_south  = 0.5 * (v[jno,icv] + v[jno-1,icv])  
                vf_east   = 0.5 * (v[jno,icv]+v[jno,icv+1])                
                vf_west   = 0.5 * (v[jno,icv]+v[jno,icv-1])
                uf_east   = 0.5 * (u[jcv,ino+1]+u[jcv-1,ino+1]) 
                uf_west   = 0.5 * (u[jcv,ino]+u[jcv-1,ino])                
           
            # Convective Part of H 
            F_c = (vf_north**2-vf_south**2+uf_east*vf_east-uf_west*vf_west )/h

            # compute diffusive fluxes 
            if (jcv==0):  # south boundary of the domain
                if ino == 0: # south west corner 
                    vcv_east  = v[jno,icv+1]
                    vcv_west  = 2.0*vbc_east - v[jno,icv] 
                    vcv_south = 0.0
                    vcv_north = v[jno+1,icv]
            
                elif ino ==  ncv_x-1: # south east  corner 
                    vcv_east  = 2.0*vbc_west - v[jno,icv] 
                    vcv_west  = v[jno,icv-1]
                    vcv_south = 0.0
                    vcv_north = v[jno+1,icv]
                    
                else: # remainder or south boundary 
                    vcv_east  = v[jno,icv+1]
                    vcv_west  = v[jno,icv-1]
                    vcv_south = 0.0 
                    vcv_north = v[jno+1,icv]
            
            elif (jcv== nno_y -1):                 
                if ino == 0: # north west corner 
                    vcv_east  = v[jno,icv+1]
                    vcv_west  = 2.0*vbc_west - v[jno,icv]
                    vcv_south = v[jno-1,icv]
                    vcv_north = 0.0 
            
                elif ino ==  ncv_x-1: # north east corner 
                    vcv_east  = 2.0*vbc_east - v[jno,icv] 
                    vcv_west  = v[jno,icv-1]
                    vcv_south = v[jno-1,icv]
                    vcv_north = 0.0 
            
                else: # remainder or south boundary 
                    vcv_east  = v[jno,icv+1]
                    vcv_west  = v[jno,icv-1]
                    vcv_south = v[jno-1,icv]
                    vcv_north = 0.0 
            
            elif (ino == 0) and (jcv > 0): # west wall besides cordners 
                vcv_east  = v[jno,icv+1]
                vcv_west  = 2.0*vbc_west - v[jno,icv]
                vcv_south = v[jno-1,icv]    
                vcv_north = v[jno+1,icv]
            
            elif (ino ==  ncv_x-1) and (jcv > 0) :  # east border besides corners   
                vcv_east  = v[jno,icv-1]
                vcv_west  = 2.0*vbc_east - v[jno,icv] 
                vcv_south = v[jno-1,icv]   
                vcv_north = v[jno+1,icv]
            else:     
                vcv_north = v[jno+1,icv]
                vcv_south = v[jno-1,icv]                
                vcv_west  = v[jno,icv-1]
                vcv_east  = v[jno,icv+1]
                    
            F_d = (mu/(h**2))*(vcv_east+vcv_west+vcv_north+vcv_south-4.0*v[jno,icv])
                
            # Compute temperature forcing function 
            F_t = -beta*(T[jcv,icv]-Tref)*gy

            # update the temporary velocity
            vt[jno,icv] = v[jno,icv] + dt * (-F_c+F_d+F_t )
            
    return
 
 
def solve_t(ncv_x,ncv_y,nno_x,nno_y,h,dt,ubc_south,ubc_north,vbc_west,vbc_east,u,v,ut,vt,k,rhoref,Cp,T,Tleft,Tright,Tref):
    # update the temporary x-velocity field (only internal u-cells)    
    Tt = np.zeros_like(T)
    
    # update the temporary x-velocity field (only internal u-cells)    
    for jcv in range(ncv_y): 
        for ino in range(nno_x):
            icv = ino
            jno = jcv
            # compute convective fluxes
            # compute x-velocity at the faces of the u-control volume &
            # compute y-velocity at the y-faces of the u-control volume 
            if (jcv==0): # south boundary of the domain
                if ino == 0:  # south west corner 
                    uf_north = 0.0 
                    uf_south = ubc_south
                    uf_east  = 0.5 * (u[jcv,ino] + u[jcv,ino+1])
                    uf_west  = 0.0 
                    vf_south = vbc_west 
                    vf_north = vbc_west   
                    Tcv_west  = 2*Tleft - T[jcv,icv] # extrapolating with T @ east boundary = Tr
                    Tcv_nwest = 2*Tleft - T[jcv+1,icv] # extrapolating with T @ north-east boundary = Tr
                    Tcv_south = T[jcv,icv] # adiabatic bc on bottom wall
                    Tf_north  = 0.25 * (T[jcv,icv]+T[jcv+1,icv]+Tcv_west+Tcv_nwest)
                    Tf_south  = 0.33 * (T[jcv,icv]+Tcv_west+Tcv_south) # average of 3 cells (2 ghost) at the bottom right corner
                    Tf_east   = T[jcv,icv+1]  #T[jcv,icv+1]   T[jcv,icv+1]
                    Tf_west   = Tcv_west   
                    
                if ino == nno_x-1:  # south west corner 
                    uf_north = 0.0 
                    uf_south = ubc_south
                    uf_east  = 0.0 
                    uf_west  = 0.5 * (u[jcv,ino] + u[jcv,ino-1])
                    vf_south = vbc_east 
                    vf_north = vbc_east
                    Tcv_east = 2*Tright - T[jcv,icv] # extrapolating with T @ east boundary = Tr
                    Tcv_neast = 2*Tright - T[jcv+1,icv] # extrapolating with T @ north-east boundary = Tr
                    Tcv_south = T[jcv,icv] # adiabatic bc on bottom wall                    
                    Tf_north = 0.25 * (T[jcv,icv]+T[jcv+1,icv]+Tcv_east+Tcv_neast)
                    Tf_south = 0.33 * (T[jcv,icv]+Tcv_east+Tcv_south) # average of 3 cells (2 ghost) at the bottom right corner
                    Tf_east  = Tcv_east
                    Tf_west  = T[jcv,icv]                    
                    
                else: #remainder of south wall 
                    uf_north = 0.5 * (v[jno+1,icv] + v[jno+1,icv-1]) 
                    uf_south = ubc_south
                    uf_east  = 0.5 * (u[jcv,ino] + u[jcv,ino+1])
                    uf_west  = 0.5 * (u[jcv,ino] + u[jcv,ino-1])
                    vf_south = 0.0  
                    vf_north = 0.5 * (v[jno+1,icv] + v[jno+1,icv-1])
                    Tcv_south = T[jcv,icv]
                    Tcv_seast = T[jcv,icv+1]
                    Tf_north = 0.25 * (T[jcv,icv]+T[jcv+1,icv]+T[jcv+1,icv+1]+T[jcv,icv+1])
                    Tf_south = 0.25 * (T[jcv,icv]+Tcv_south+Tcv_seast+T[jcv,icv+1])
                    Tf_east  = T[jcv,icv+1]
                    Tf_west  = T[jcv,icv]                        
            elif (jcv==ncv_y-1): # north boundary of the domain
                if ino == 0:  # north west corner 
                    uf_north = ubc_north
                    uf_south = 0.0 
                    uf_east  = 0.5 * (u[jcv,ino] + u[jcv,ino+1])
                    uf_west  = 0.0 
                    vf_south = vbc_west 
                    vf_north = vbc_west 
                    Tcv_west  = 2*Tleft - T[jcv,icv]   # extrapolating with T @ west boundary = Tr
                    Tcv_swest = 2*Tleft - T[jcv-1,icv] # extrapolating with T @ north-west boundary = Tr
                    Tcv_north = T[jcv,icv] # adiabatic bc on top wall  
                    Tf_north = 0.33 * (T[jcv,icv]+Tcv_north+Tcv_east)                    
                    Tf_south = 0.25 * (T[jcv,icv]+T[jcv-1,icv]+Tcv_swest+Tcv_west)   
                    Tf_east  = T[jcv,icv+1]
                    Tf_west  = Tcv_west   
                    
                if ino == nno_x-1:  # north east corner 
                    uf_north = ubc_north
                    uf_south = 0.0 
                    uf_east  = 0.0 
                    uf_west  = 0.5 * (u[jcv,ino] + u[jcv,ino-1])
                    vf_south = vbc_east 
                    vf_north = vbc_east
                    Tcv_east  = 2*Tright - T[jcv,icv] # extrapolating with T @ east boundary = Tr
                    Tcv_seast = 2*Tright - T[jcv-1,icv] # extrapolating with T @ north-east boundary = Tr
                    Tcv_north = T[jcv,icv] # adiabatic bc on top wall 
                    Tf_north = 0.33 * (T[jcv,icv]+Tcv_north+Tcv_east)
                    Tf_south = 0.25 * (T[jcv,icv]+T[jcv-1,icv]+Tcv_seast+Tcv_east)  
                    Tf_east  = Tcv_east
                    Tf_west  = T[jcv,icv] 
                    
                else: # remainder of north wall 
                    uf_north = ubc_north
                    uf_south = 0.5 * (v[jno-1,icv] + v[jno-1,icv-1]) 
                    uf_east  = 0.5 * (u[jcv,ino] + u[jcv,ino+1])
                    uf_west  = 0.5 * (u[jcv,ino] + u[jcv,ino-1])
                    vf_south = 0.5 * (v[jno-1,icv] + v[jno-1,icv-1])        
                    vf_north = 0.0   
                    Tcv_neast = T[jcv,icv+1] # adiabatic on top wall
                    Tcv_north = T[jcv,icv] # adiabatic bc on top wall
                    Tf_north = 0.25 * (T[jcv,icv]+T[jcv,icv+1]+Tcv_north+Tcv_neast)
                    Tf_south = 0.25 * (T[jcv,icv]+T[jcv-1,icv]+T[jcv-1,icv+1]+T[jcv,icv+1]) 
                    Tf_east  = T[jcv,icv+1]
                    Tf_west  = T[jcv,icv]                       
            elif (ino == 0) and (jcv > 0) :  # west border besides corners 
                    uf_north = 0.0 
                    uf_south = 0.0 
                    uf_east  = 0.5 * (u[jcv,ino] + u[jcv,ino+1])
                    uf_west  = 0.0 
                    vf_south = vbc_west 
                    vf_north = vbc_west  
                    Tcv_west  = 2*Tleft - T[jcv,icv] # extrapolating with T @ east boundary = Tr
                    Tcv_north = 2*Tleft - T[jcv+1,icv] # extrapolating with T @ north-east boundary = Tr
                    Tcv_south = 2*Tleft - T[jcv-1,icv]
                    Tcv_east  = T[jcv,icv]
                    Tcv_neast = T[jcv+1,icv]
                    Tcv_seast = T[jcv-1,icv] 
                    Tf_north  = 0.25 * (Tcv_neast + Tcv_north +Tcv_west +Tcv_east )
                    Tf_south  = 0.25 * (Tcv_seast + Tcv_south +Tcv_west +Tcv_east ) 
                    Tf_east   = T[jcv,icv]   
                    Tf_west   = Tcv_west  
                    
            elif (ino == nno_x-1) and (jcv > 0) :  # east border besides corners 
                    uf_north = 0.0 
                    uf_south = 0.0 
                    uf_east  = 0.0 
                    uf_west  = 0.5 * (u[jcv,ino] + u[jcv,ino-1])
                    vf_south = vbc_east 
                    vf_north = vbc_east  
                    Tcv_east  = 2*Tright - T[jcv,icv] # extrapolating with T @ east boundary = Tr
                    Tcv_north = 2*Tright - T[jcv+1,icv] # extrapolating with T @ north-east boundary = Tr
                    Tcv_south = 2*Tright - T[jcv-1,icv]
                    Tcv_west  = T[jcv,icv-1]
                    Tcv_nwest = T[jcv+1,icv-1]
                    Tcv_swest = T[jcv-1,icv-1] 
                    Tf_north  = 0.25 * ( Tcv_north +Tcv_nwest + Tcv_west +  Tcv_east )
                    Tf_south  = 0.25 * (Tcv_south +Tcv_swest + Tcv_west +  Tcv_east ) 
                    Tf_west   = T[jcv,icv]   
                    Tf_east   = Tcv_east  
                    
            else: # remainder of computational domain
                uf_north = 0.5 * (u[jcv,ino]   + u[jcv+1,ino])              
                uf_south = 0.5 * (u[jcv,ino]   + u[jcv-1,ino])
                uf_east  = 0.5 * (u[jcv,ino]   + u[jcv,ino+1])
                uf_west  = 0.5 * (u[jcv,ino]   + u[jcv,ino-1])  
                vf_south = 0.5 * (v[jno,icv]   + v[jno,icv-1])
                vf_north = 0.5 * (v[jno+1,icv] + v[jno+1,icv-1])
              
            # Convective Part of H        
            F_c = (uf_east*Tf_east-uf_west*Tf_west+vf_north*Tf_north-vf_south*Tf_south)/h
                
            # compute diffusive fluxes
            if (jcv==0):  # south boundary of the domain
                if ino == 1: # south west corner 
                    ucv_east  = u[jcv,ino+1]
                    ucv_west  = 0.0                
                    ucv_south = 2.0 * ubc_south - u[jcv,ino] # half at corner # extrapolating with u @ south boundary = 0
                    ucv_north = u[jcv+1,ino]
                    Tcv_east  = T[jcv,icv+1]
                    Tcv_west  = 2*Tleft - T[jcv,icv]
                    Tcv_south = T[jcv,icv] # extrapolating with T @ south boundary (adiabatic)
                    Tcv_north = T[jcv+1,icv] 
                    
                elif ino == nno_x-2: # south east corner 
                    ucv_east  = 0.0
                    ucv_west  = u[jcv,ino-1]                
                    ucv_south = 2.0 * ubc_south - u[jcv,ino] # half at corner # extrapolating with u @ south boundary = 0
                    ucv_north = u[jcv+1,ino]                    
                    Tcv_east = 2*Tright - T[jcv,icv]
                    Tcv_west  = T[jcv,icv-1]
                    Tcv_south = T[jcv,icv] # extrapolating with T @ south boundary (adiabatic)
                    Tcv_north = T[jcv+1,icv] 
                    
                elif (ino > 0) and (ino < nno_x-2): # remainder or south boundary 
                    ucv_east  = u[jcv,ino+1]
                    ucv_west  = u[jcv,ino-1]                
                    ucv_south = 2.0 * ubc_south - u[jcv,ino] # extrapolating with u @ south boundary = 0
                    ucv_north = u[jcv+1,ino]  
                    Tcv_east  = T[jcv,icv+1]
                    Tcv_west  = T[jcv,icv-1]
                    Tcv_south = T[jcv,icv]   # extrapolating with T @ south boundary (adiabatic)
                    Tcv_north = T[jcv+1,icv]
                        
            elif (jcv==ncv_y-1):                 
                if ino == 1: # north west corner 
                    ucv_east  = u[jcv,ino+1]
                    ucv_west  = 0.0                
                    ucv_south = u[jcv-1,ino] 
                    ucv_north = 2.0 * ubc_north - u[jcv,ino]# half at corner # extrapolating with u @ south boundary = 0    
                    Tcv_east  = T[jcv,icv+1]
                    Tcv_west  = 2*Tleft - T[jcv,icv]
                    Tcv_south = T[jcv-1,icv]# extrapolating with T @ south boundary (adiabatic)
                    Tcv_north = T[jcv,icv] 
                    
                elif ino == nno_x-2: # north east corner 
                    ucv_east  = 0.0
                    ucv_west  = u[jcv,ino-1]                
                    ucv_south = u[jcv-1,ino]  
                    ucv_north = 2.0 * ubc_north - u[jcv,ino] # half at corner # extrapolating with u @ south boundary = 0
                    Tcv_east =  2*Tright - T[jcv,icv]
                    Tcv_west  = T[jcv,icv-1]
                    Tcv_south = T[jcv-1,icv]  # extrapolating with T @ south boundary (adiabatic)
                    Tcv_north = T[jcv,icv]                          
                
                elif (ino > 0) and (ino < nno_x-2):# remainder or north boundary 
                    ucv_east  = u[jcv,ino+1]
                    ucv_west  = u[jcv,ino-1]                
                    ucv_south = u[jcv-1,ino]    # extrapolating with u @ south boundary = 0
                    ucv_north = 2.0 * ubc_north - u[jcv,ino] 
                    Tcv_east  = T[jcv,icv+1]
                    Tcv_west  = T[jcv,icv-1]
                    Tcv_south = T[jcv-1,icv]
                    Tcv_north = T[jcv,icv]   
             
            elif (ino == 0) and (jcv > 1): # west wall besides cordners  
                ucv_east  = u[jcv,ino+1]  
                ucv_west  = 0.0              
                ucv_south = u[jcv-1,ino]  
                ucv_north = u[jcv+1,ino]  
                Tcv_west  = 2*Tleft - T[jcv,icv] # extrapolating with T @ west boundary = Tl
                Tcv_east  = T[jcv,icv+1]
                Tcv_south = T[jcv-1,icv]
                Tcv_north = T[jcv+1,icv]
                    
            elif (ino == nno_x-1) and (jcv > 1) :  # east border besides corners   
                ucv_east  = 0.0
                ucv_west  = u[jcv,ino-1]
                ucv_south = u[jcv-1,ino]
                ucv_north = u[jcv+1,ino]
                Tcv_west  = T[jcv,icv-1]
                Tcv_east  = 2*Tright - T[jcv,icv] # extrapolating with T @ west boundary = Tr
                Tcv_south = T[jcv-1,icv]
                Tcv_north = T[jcv+1,icv]
                
            else: #remainder of domain 
                ucv_north = u[jcv+1,ino]                
                ucv_south = u[jcv-1,ino]
                ucv_east  = u[jcv,ino+1]
                ucv_west  = u[jcv,ino-1]     
                Tcv_west  = T[jcv,icv-1]
                Tcv_east  = T[jcv,icv+1] 
                Tcv_south = T[jcv-1,icv]
                Tcv_north = T[jcv+1,icv]
                
            # Diffusive part of H - second order central difference       
            F_d = (k/rhoref/Cp/(h**2))*(Tcv_east+Tcv_west+Tcv_north+Tcv_south-4.0*T[jcv,ino])
            
            # update the temperature 
            Tt[jcv,icv] = T[jcv,icv] + dt * ( -F_c+F_d )
                
    ## update the temporary y-velocity field (only internal v-cells)    
    #for jno in range (nno_y): 
        #for icv in range(ncv_x):
            #ino = icv
            #jcv = jno
            ## compute convective fluxes 
            ## compute y-velocity at the faces of the v-control volume &
            ## compute x-velocity at the x-faces of the v-control volume 
            #if (jcv==0): # south boundary of the domain
                #if ino == 0: # south east corner 
                    #vf_north  = 0.5 * (v[jno,icv] + v[jno+1,icv])
                    #vf_south  = 0.0                   
                    #vf_east   = 0.5 * (v[jno,icv]+v[jno,icv+1])
                    #vf_west   = vbc_west
                    #uf_west   = ubc_south
                    #uf_east   = ubc_south                    
                #if ino == nno_x-1: # south west corner 
                    #vf_north  = 0.5 * (v[jno,icv] + v[jno+1,icv])
                    #vf_south  = 0.0                    
                    #vf_east   = vbc_east 
                    #vf_west   = 0.5 * (v[jno,icv]+v[jno,icv-1]) 
                    #uf_west   = ubc_south
                    #uf_east   = ubc_south  
                #else: # remainder of south wall 
                    #vf_north  = 0.5 * (v[jno,icv] + v[jno+1,icv])
                    #vf_south  = 0.0                     
                    #vf_east   = 0.5 * (v[jno,icv]+v[jno,icv+1])
                    #vf_west   = 0.5 * (v[jno,icv]+v[jno,icv-1])  
                    #uf_west   = ubc_south
                    #uf_east   = ubc_south
                    
            #elif (jcv==ncv_y-1): # north boundary of the domain
                #if ino == 0: # north east corner 
                    #vf_north  = 0.0
                    #vf_south  = 0.5 * (v[jno,icv] + v[jno-1,icv])                 
                    #vf_east   = 0.5 * (v[jno,icv]+v[jno,icv+1])
                    #vf_west   = vbc_west
                    #uf_west   = ubc_north
                    #uf_east   = ubc_north                   
                #if ino == nno_x-1: # north west corner 
                    #vf_north  = 0.0   
                    #vf_south  = 0.5 * (v[jno,icv] + v[jno-1,icv])                  
                    #vf_east   = vbc_east 
                    #vf_west   = 0.5 * (v[jno,icv]+v[jno,icv-1]) 
                    #uf_west   = ubc_north
                    #uf_east   = ubc_north  
                #else: # remainder of north wall 
                    #vf_north  = 0.0
                    #vf_south  = 0.5 * (v[jno,icv] + v[jno-1,icv])                   
                    #vf_east   = 0.5 * (v[jno,icv]+v[jno,icv+1])
                    #vf_west   = 0.5 * (v[jno,icv]+v[jno,icv-1])  
                    #uf_west   = ubc_north
                    #uf_east   = ubc_north                 
                 
            #elif (icv == 0) and (jcv > 0) :  # west border besides corners 
                #vf_north = 0.5 * (v[jno,icv] + v[jno+1,icv+1]) 
                #vf_south = 0.5 * (v[jno,icv] + v[jno+1,icv-1]) 
                #vf_east  = 0.5 * (v[jno,icv] + v[jno,icv+1])
                #vf_west  = vbc_west     
                #uf_east  = 0.5 * (u[jno,icv] + u[jno,icv+1])   
                #uf_west  = 0.0  
        
            #elif (icv == nno_x-1) and (jcv > 0) :  # east border besides corners 
                #vf_north = 0.5 * (v[jno,icv] + v[jno+1,icv+1])
                #vf_south = 0.5 * (v[jno,icv] + v[jno+1,icv-1])
                #vf_east  = vbc_east 
                #vf_west  = 0.5 * (v[jno,icv] + v[jno,icv-1])    
                #uf_east  = 0.0 
                #uf_west  = 0.5 * (u[jno,icv] + u[jno,icv-1])
            
            #else: 
                #vf_north  = 0.5 * (v[jno,icv] + v[jno+1,icv])
                #vf_south  = 0.5 * (v[jno,icv] + v[jno-1,icv])  
                #vf_east   = 0.5 * (v[jno,icv]+v[jno,icv+1])                
                #vf_west   = 0.5 * (v[jno,icv]+v[jno,icv-1])
                #uf_east   = 0.5 * (u[jcv,ino+1]+u[jcv-1,ino+1]) 
                #uf_west   = 0.5 * (u[jcv,ino]+u[jcv-1,ino])                
           
            ## Convective Part of H 
            #F_c = (vf_north**2-vf_south**2+uf_east*vf_east-uf_west*vf_west )/h

            ## compute diffusive fluxes 
            #if (jcv==0):  # south boundary of the domain
                #if ino == 0: # south west corner 
                    #vcv_east  = v[jno,icv+1]
                    #vcv_west  = 2.0*vbc_east - v[jno,icv] 
                    #vcv_south = 0.0
                    #vcv_north = v[jno+1,icv]
            
                #elif ino == nno_x-1: # south east  corner 
                    #vcv_east  = 2.0*vbc_west - v[jno,icv] 
                    #vcv_west  = v[jno,icv-1]
                    #vcv_south = 0.0
                    #vcv_north = v[jno+1,icv]
                    
                #else: # remainder or south boundary 
                    #vcv_east  = v[jno,icv+1]
                    #vcv_west  = v[jno,icv-1]
                    #vcv_south = 0.0 
                    #vcv_north = v[jno+1,icv]
            
            #elif (jcv==ncv_y-1):                 
                #if ino == 0: # north west corner 
                    #vcv_east  = v[jno,icv+1]
                    #vcv_west  = 2.0*vbc_west - v[jno,icv]
                    #vcv_south = v[jno-1,icv]
                    #vcv_north = 0.0 
            
                #elif ino == nno_x-1: # north east corner 
                    #vcv_east  = 2.0*vbc_east - v[jno,icv] 
                    #vcv_west  = v[jno,icv-1]
                    #vcv_south = v[jno-1,icv]
                    #vcv_north = 0.0 
            
                #else: # remainder or south boundary 
                    #vcv_east  = v[jno,icv+1]
                    #vcv_west  = v[jno,icv-1]
                    #vcv_south = v[jno-1,icv]
                    #vcv_north = 0.0 
            
            #elif (ino == 0) and (jcv > 0): # west wall besides cordners 
                #vcv_east  = v[jno,icv+1]
                #vcv_west  = 2.0*vbc_west - v[jno,icv]
                #vcv_south = v[jno-1,icv]    
                #vcv_north = v[jno+1,icv]
            
            #elif (ino == nno_x-1) and (jcv > 0) :  # east border besides corners   
                #vcv_east  = v[jno,icv-1]
                #vcv_west  = 2.0*vbc_east - v[jno,icv] 
                #vcv_south = v[jno-1,icv]   
                #vcv_north = v[jno+1,icv]
            #else:     
                #vcv_north = v[jno+1,icv]
                #vcv_south = v[jno-1,icv]                
                #vcv_west  = v[jno,icv-1]
                #vcv_east  = v[jno,icv+1]
                    
            #F_d = (mu/(h**2))*(vcv_east+vcv_west+vcv_north+vcv_south-4.0*v[jno,icv])
                
            ## Compute temperature forcing function 
            #F_t = -beta*(T[jcv,icv]-Tref)*gy

            ## update the temporary velocity
            #vt[jno,icv] = v[jno,icv] + dt * (-F_c+F_d+F_t )
    
    T = Tt 
    
    return T

def correct_uv(ncv_x,ncv_y,nno_x,nno_y,h,dt,u,v,ut,vt,p):
    # correct (and update) the velocity for the internal cells
    for jcv in range(ncv_y): 
        for ino in range(1,nno_x-1):
            u[jcv,ino] = ut[jcv,ino] - (dt/h)*(p[jcv,ino]-p[jcv,ino-1])   
    for jno in range (1,nno_y-1): 
        for icv in range(ncv_x):
            v[jno,icv] = vt[jno,icv] - (dt/h)*(p[jno,icv]-p[jno-1,icv]) 
    return

def plot_uv(ncv_x,ncv_y,nno_x,nno_y,ubc_south,ubc_north,vbc_west,vbc_east,x,y,u,v):
    # interpolate velocity on nodes 
    u_plot = np.zeros((nno_y,nno_x)) 
    v_plot = np.zeros((nno_y,nno_x)) 
    vmag_plot = np.zeros((nno_y,nno_x))
    for jno in range(nno_y):
        u_plot[jno,0] = 0.0
        v_plot[jno,0] = vbc_west
        u_plot[jno,nno_x-1] = 0.0
        v_plot[jno,nno_x-1] = vbc_east
    for ino in range(1,nno_x-1):
        u_plot[0,ino] = ubc_south
        v_plot[0,ino] = 0.0
        u_plot[nno_y-1,ino] = ubc_north
        v_plot[nno_y-1,ino] = 0.0
    for jno in range(1,nno_y-1):
        for ino in range(1,nno_x-1):  
            u_plot[jno,ino] = 0.5*(u[jno,ino]+u[jno-1,ino])
            v_plot[jno,ino] = 0.5*(v[jno,ino]+v[jno,ino-1])
    # compute velocity magnitude
    for jno in range(nno_y):
        for ino in range(nno_x):  
            vmag_plot[jno,ino] = np.sqrt(u_plot[jno,ino]*u_plot[jno,ino]+v_plot[jno,ino]*v_plot[jno,ino])
   
    # plot vontours of velocity magnitude and velocity vectors
    X, Y = np.meshgrid(x, y)
    fig = pl.figure(figsize=(11, 7), dpi=100)
    pl.contourf(X, Y, vmag_plot, alpha=0.8, cmap=cm.jet, levels = 50)
    pl.gca().set_aspect('equal')
    pl.colorbar()
    pl.contour(X, Y, vmag_plot, cmap=cm.jet)
    pl.quiver(X , Y , u_plot , v_plot )
    pl.xlabel('X')
    pl.ylabel('Y');
    pl.title('U,V');
    
    return

def plot_t(ncv_x,ncv_y,nno_x,nno_y,ubc_south,ubc_north,vbc_west,vbc_east,x,y,T):
    Tmag_plot = np.zeros((nno_y,nno_x))
    # compute temperature
    for jcv in range(ncv_y):
        for icv in range(ncv_x):  
            Tmag_plot[jcv,icv] = T[jcv,icv]
   
    # plot vontours of velocity magnitude and velocity vectors
    X, Y = np.meshgrid(x, y)
    fig = pl.figure(figsize=(11, 7), dpi=100)
    pl.contourf(X, Y, Tmag_plot, alpha=0.8, cmap=cm.jet, levels = 50)
    pl.gca().set_aspect('equal')
    pl.colorbar()
    pl.contour(X, Y, Tmag_plot, cmap=cm.jet)
    pl.xlabel('X')
    pl.ylabel('Y');
    pl.title('T');
    
    return

def compute_tau(ncv_x,ncv_y,nno_x,nno_y,h,ubc_south,ubc_north,vbc_west,vbc_east,u,v,mu):
    # compute total friction on all walls
    tau_south = 0.0
    tau_north = 0.0
    tau_east = 0.0
    tau_west = 0.0
    for ino in range(1,nno_x-1):
        tau = mu*(u[0,ino]-ubc_south)/(0.5*h)
        tau_south = tau_south+tau*h
        tau = mu*(ubc_north-u[ncv_y-1,ino])/(0.5*h)
        tau_north = tau_north+tau*h
    for jno in range(1,nno_y-1):
        tau = mu*(v[jno,0]-vbc_west)/(0.5*h)
        tau_west = tau_west+tau*h
        tau = mu*(vbc_east-v[jno,ncv_x-1])/(0.5*h)
        tau_east = tau_east+tau*h
    print ("Friction S/N/W/E",tau_south,tau_north,tau_west,tau_east)

    return tau_east

# solve the Navier-Stokes Equations
def solve(ncv,ubc_south,ubc_north,vbc_west,vbc_east,solver,plot,mu,Tref,Tleft,Tright,beta,gx,gy,Cp,k,rhoref,deltat,timesteps):  # Input temperature parameters 
    
    #assume uniform grid in x&y in the unit square [0:1]x[0:1]
    ncv_x = ncv        # addition of 1 
    ncv_y = ncv_x    # addition of 1 
    
    # define grid
    nno_x = ncv_x + 2
    nno_y = ncv_y + 2

    # initialize
    u = np.zeros((ncv_y+1, nno_x))   # u velocity field
    v = np.zeros((nno_y , ncv_x+1))   # v velocity field
    p = np.zeros((nno_y , nno_x ))  # pressure   
    T = np.ones((nno_y  , nno_x ))* Tright     
    ut = np.zeros_like(u)  # temporary u velocity field
    vt = np.zeros_like(v)  # temporary v velocity field

    # time step and grid size
    h=1.0/ncv_x
    vmax = max(abs(ubc_north),abs(ubc_south),abs(vbc_west),abs(vbc_east)) #If nt walls moving, vmax = 0
    # check stabbility limit
    dt = deltat              # defaults to 0.02
    dt = min(dt,.2*h*h/mu)   # viscous limit
    #dt = min(dt,2.0*mu/vmax) # Peclet number limit

    # grid points (nodes)
    x = np.linspace(0.0,1.0,nno_x)
    y = np.linspace(0.0,1.0,nno_y)

    # initialize the matrix of coefficient
    a_east  = np.zeros((ncv_y,ncv_x)) 
    a_west  = np.zeros((ncv_y,ncv_x)) 
    a_north = np.zeros((ncv_y,ncv_x)) 
    ### pentadiagonal LHS of Poisson solver (holds lap(p) coeffs.)
    a_south = np.zeros((ncv_y,ncv_x))  
    a_p     = np.zeros((ncv_y,ncv_x)) 
    b       = np.zeros((ncv_y,ncv_x)) 
    #will hold div(ut) at each cell center for Poisson solver RHS
    phi     = np.zeros((ncv_y,ncv_x)) 

    # build the matrix for the pressure equation
    for jcv in range(ncv_y):
        for icv in range(ncv_x): 
            if (jcv>0):
                a_south[jcv,icv] = 1.0
            if (jcv<ncv_y-1):
                a_north[jcv,icv] = 1.0
            if (icv>0):
                a_west[jcv,icv] = 1.0
            if (icv<ncv_x-1):
                a_east[jcv,icv] = 1.0
            a_p[jcv,icv] = -(a_north[jcv,icv]+a_south[jcv,icv]+a_east[jcv,icv]+a_west[jcv,icv]) 
        

    
    # time loop
    time = 0.0
    for istep in range(timesteps):
        #update the temporary velocity field
        solve_uv(ncv_x,ncv_y,nno_x,nno_y,h,dt,ubc_south,ubc_north,vbc_west,vbc_east,u,v,ut,vt,mu,p,beta,Tref,gx,gy,T)
    
        # solve for temperature 
        new_T = solve_t(ncv_x,ncv_y,nno_x,nno_y,h,dt,ubc_south,ubc_north,vbc_west,vbc_east,u,v,ut,vt,k,rhoref,Cp,T,Tleft,Tright,Tref)
        
        #compute the pressure field
        solve_p(ncv_x,ncv_y,nno_x,nno_y,h,dt,p,a_east,a_west,a_north,a_south,a_p,b,ut,vt,"ilu")
        
        #correct the velocity field
        correct_uv(ncv_x,ncv_y,nno_x,nno_y,h,dt,u,v,ut,vt,p)
        
        # advance time
        time = time+dt
    if plot:
        # plot results reporting velocity on all nodes (average)
        plot_uv(ncv_x,ncv_y,nno_x,nno_y,ubc_south,ubc_north,vbc_west,vbc_east,x,y,u,v)
        # plot results reporting temperature on all nodes (average) this function needs to be written
        
        x = np.linspace(0.0,1.0,ncv_y)
        y = np.linspace(0.0,1.0,ncv_y)    
        X,Y= np.meshgrid(x , y)   
        fig = pl.figure()  
        axes1 = fig.add_subplot(1,1,1)    
        CS_1 = axes1.contourf(X,Y,new_T, 100 , cmap=cm.jet  )     
        cbar1 = fig.colorbar(CS_1, ax = axes1)   
        cbar1.set_label('Temerature')  
        
    #compute and write skin friction
    tau_east = compute_tau(ncv_x,ncv_y,nno_x,nno_y,h,ubc_south,ubc_north,vbc_west,vbc_east,u,v,mu)
    
    return tau_east



 # grid convergence
# monitor friction coefficient on East wall (tau_east) with successive refinements
def converge(levels):
    tau      = np.zeros(levels)

    for i in range(levels):
        tau[i] = solve(4*(2**i),0.0,1.0,0.0,0.0,"ilu",0)  
    error    = (tau[1:] - tau[0:-1])/tau[1:]*100  #relative error in tau_east
    l        = np.linspace(1,levels-1,levels-1)
    lSpl     = np.linspace(1,len(l),300)
    spl      = make_interp_spline(l, error, k=2)  # smooth out curve with a b-spline
    errorSpl = spl(lSpl)
    fig1 = pl.figure(figsize=(4,3), dpi=100)
    pl.plot(lSpl, errorSpl)
    pl.xlabel('number of CVs')
    pl.ylabel('% error in east wall friction')
    pl.tick_params(
        axis='x',          
        which='both',      
        bottom=False,      
        top=False)
    pl.xticks([], [])
    return error

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    pl.show()
