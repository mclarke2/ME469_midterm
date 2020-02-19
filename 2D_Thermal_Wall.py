#set things up
import os
import numpy  as np                       
import time, sys  
import matplotlib.pyplot as pl
from matplotlib import cm
from scipy.interpolate import make_interp_spline, BSpline
import pdb
 
def main():
    Ra        = 10^4      # set the Rayleigh number, viscosity will be a free parameter to match it
    Tref      = 273       # reference Temperature (K)
    Tleft     = 470       # left (hot) dirichlet boundary condition on temperature
    Tright    = 109       # right (cold) dirichlet boundary condition on temperature
    beta      = 0.00369   # thermal expansion coefficient (1/K)  0.00369 for air at 0 deg C, 1 Bar 
    gmag      = 9.81      # gravitational acceleration magnitude
    gx        = 0*gmag    # gravitational acceleration in x direction
    gy        = -1*gmag   # gravitational acceleration in y direction
    Cp        = 1004.703  # specific heat capacity (J/kg/K), 1000 for air at stp
    k         = 0.0246    # thermal conductivity (W/m/K), 0.026 for air
    rhoref    = 1.225     # reference density (kg/m^3), 1.225 for air at stp
    
    # this is a flag for constant temperature (lid driven)
    if Tleft == Tright:   
        mu = 0.1
        ubc_north = 1.0  # velocity of north wall
    else:  
        mu = gmag*beta*k/Cp/rhoref*abs((Tleft-Tright))/Ra     # diffusion coefficient, really nu [m^2/s], assumes x = 1 
        ubc_north = 0.0  # velocity of north wall
    
    deltat    = 0.0001    # time  step (stability limit checked later)
    timesteps = 30000  # number of timesteps
    solver    = "ilu"
    plot      = 1
    ncv       = 16   # 50
    ubc_south = 0.0  # velocity of south wall 
    vbc_west  = 0.0  # velocity of west  wall
    vbc_east  = 0.0  # velocity of east  wall
    
    solve(ncv,ubc_south,ubc_north,vbc_west,vbc_east,solver,plot,mu,Tref,Tleft,Tright,beta,gx,gy,Cp,k,rhoref,deltat,timesteps)
    
    return

def clear():
    os.system( 'clear' )

# solve a linear system using gauss-seidel
def solve_gs(ncv_x,ncv_y,dx,dy,phi,a_east,a_west,a_north,a_south,a_p,b,n_iter,tolerance):

    res = np.zeros((ncv_y,ncv_x)) #initialize

    for n in range(n_iter):
        # update solution - internal cells
        for jcv in range(ncv_y):
            jcv_north = min(jcv+1,ncv_y-1)      # assumes a_north[ncv_y-1,icv]=0
            jcv_south = max(jcv-1,0)            # assumes a_south[0,icv]=0
            for icv in range(ncv_x):
                icv_east = min(icv+1,ncv_x-1)   # assumes a_east[jcv,ncv_x-1]=0
                icv_west = max(icv-1,0)         # assumes a_west[jcv,0]=0
                phi[jcv,icv]=(b[jcv,icv]
                              -a_east [jcv,icv]*phi[jcv,icv_east ]-a_west [jcv,icv]*phi[jcv,icv_west ] 
                          -a_north[jcv,icv]*phi[jcv_north,icv]-a_south[jcv,icv]*phi[jcv_south,icv])/a_p[jcv,icv]

        for jcv in range(ncv_y):
            jcv_north = min(jcv+1,ncv_y-1)      # assumes a_north[ncv_y-1,icv]=0
            icv_south = max(jcv-1,0)            # assumes a_south[0,icv]=0
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
        for ino in range(1,nno_x-1):
            icv = ino
            jno = jcv
            # compute convective fluxes
    
            # compute x-velocity at the faces of the u-control volume &
            # compute y-velocity at the y-faces of the u-control volume
            uf_east  = 0.5 * (u[jcv,ino] + u[jcv,ino+1])
            uf_west  = 0.5 * (u[jcv,ino] + u[jcv,ino-1])
            if (jcv==0): # south boundary of the domain
                uf_south = ubc_south
                vf_south = 0.0  
                uf_north = 0.5 * (u[jcv,ino]   + u[jcv+1,ino])
                vf_north = 0.5 * (v[jno+1,icv] + v[jno+1,icv-1])
            elif (jcv==ncv_y-1): # north boundary of the domain
                uf_south = 0.5 * (u[jcv,ino]   + u[jcv-1,ino])
                vf_south = 0.5 * (v[jno,icv]   + v[jno,icv-1])
                uf_north = ubc_north
                vf_north = 0.0
            else:
                #print "jcv,ino",jcv,ino
                uf_south = 0.5 * (u[jcv,ino]   + u[jcv-1,ino])
                vf_south = 0.5 * (v[jno,icv]   + v[jno,icv-1])
                uf_north = 0.5 * (u[jcv,ino]   + u[jcv+1,ino])
                vf_north = 0.5 * (v[jno+1,icv] + v[jno+1,icv-1])
    
            # Convective Part of H        
            F_c = (uf_east**2-uf_west**2+uf_north*vf_north-uf_south*vf_south)/h
    
            # compute diffusive fluxes
            ucv_east = u[jcv,ino+1]
            ucv_west = u[jcv,ino-1]
            if (jcv==0):  # south boundary of the domain
                ucv_south = 2.0 * ubc_south - u[jcv,ino] # extrapolating with u @ south boundary = 0
                ucv_north = u[jcv+1,ino]
            elif (jcv==ncv_y-1): # north boundary of the domain  u @ north boundary = u_lid
                ucv_south = u[jcv-1,ino]
                ucv_north = 2.0 * ubc_north - u[jcv,ino]
            else:
                ucv_south = u[jcv-1,ino]
                ucv_north = u[jcv+1,ino]
    
            # Diffusive part of H - second order central difference       
            F_d = (mu/(h**2))*(ucv_east+ucv_west+ucv_north+ucv_south-4.0*u[jcv,ino])
    
            # Compute temperature forcing function 
            F_t = -beta*(T[jcv,icv]-Tref)*gx
    
            # update the temporary velocity ( we added delta p )
            ut[jcv,ino] = u[jcv,ino] + dt * ( -F_c+F_d+F_t ) 
    
    # update the temporary y-velocity field (only internal v-cells)    
    for jno in range (1,nno_y-1): 
        for icv in range(ncv_x):
            ino = icv
            jcv = jno
            # compute convective fluxes
    
            # compute y-velocity at the faces of the v-control volume &
            # compute x-velocity at the x-faces of the v-control volume    
            vf_north  = 0.5 * (v[jno,icv] + v[jno+1,icv])
            vf_south  = 0.5 * (v[jno,icv] + v[jno-1,icv])
            if (icv==0): # west boundary of the domain
                uf_west = 0.0
                vf_west = vbc_west
                uf_east = 0.5 * (u[jcv,ino+1]+u[jcv-1,ino+1])
                vf_east = 0.5 * (v[jno,icv]+v[jno,icv+1])
            elif (icv==ncv_x-1): # east boundary of the domain
                uf_west = 0.5 * (u[jcv,ino]+u[jcv-1,ino])
                vf_west = 0.5 * (v[jno,icv]+v[jno,icv-1])
                uf_east = 0.0
                vf_east = vbc_east
            else:
                uf_west = 0.5 * (u[jcv,ino]+u[jcv-1,ino])
                vf_west = 0.5 * (v[jno,icv]+v[jno,icv-1])
                uf_east = 0.5 * (u[jcv,ino+1]+u[jcv-1,ino+1])
                vf_east = 0.5 * (v[jno,icv]+v[jno,icv+1])
    
            F_c = (vf_north**2-vf_south**2+uf_east*vf_east-uf_west*vf_west )/h
    
            # compute diffusive fluxes
            vcv_north = v[jno+1,icv]
            vcv_south = v[jno-1,icv]
            if (icv==0):  # west boundary of the domain
                vcv_west = 2.0*vbc_west - v[jno,icv] # extrapolating with v @ west boundary = 0
                vcv_east =   v[jno,icv+1]
            elif (icv==ncv_x-1): # north boundary of the domain
                vcv_west =   v[jno,icv-1]    
                vcv_east = 2.0*vbc_east - v[jno,icv] # extrapolating with v @ east boundary = 0
            else:
                vcv_west = v[jno,icv-1]
                vcv_east = v[jno,icv+1]
    
            F_d = (mu/(h**2))*(vcv_east+vcv_west+vcv_north+vcv_south-4.0*v[jno,icv])
    
            # Compute temperature forcing function 
            F_t = -beta*(T[jcv,icv]-Tref)*gy
    
            # update the temporary velocity
            vt[jno,icv] = v[jno,icv] + dt * ( -F_c+F_d+F_t )    

    return
 
 
def solve_t(ncv_x,ncv_y,nno_x,nno_y,h,dt,ubc_south,ubc_north,vbc_west,vbc_east,u,v,ut,vt,k,rhoref,Cp,T,Tt,Tleft,Tright,Tref):
    
    # update the temporary temperature field
    for jcv in range(ncv_y): 
        for ino in range(ncv_x):
            icv = ino
            jno = jcv
            # compute convective fluxes
            # compute x-velocity at the faces of the u-control volume &
            # compute y-velocity at the y-faces of the u-control volume

            uf_east   = u[jcv,ino+1] 
            uf_west   = u[jcv,ino]
            vf_south  = v[jno,icv] 
            vf_north  = v[jno+1,icv]              
            if (jcv==0): # south boundary of the domain 
                if ino == 0: # south west corner 
                    Tf_north  = 0.5 * (T[jcv,icv]+T[jcv+1,icv])
                    Tf_south  = T[jcv,icv]
                    Tf_east   = 0.5 * (T[jcv,icv]+T[jcv,icv+1])
                    Tf_west   = Tleft 
                    
                elif ino == ncv_x-1:  # south east corner 
                    Tf_north = 0.5 * (T[jcv,icv]+T[jcv+1,icv])
                    Tf_south = T[jcv,icv]
                    Tf_east  = Tright
                    Tf_west  = 0.5 * (T[jcv,icv]+T[jcv,icv-1])         
                    
                else: #remainder of south wall 
                    Tf_north  = 0.5 * (T[jcv,icv]+T[jcv+1,icv])  
                    Tf_south  = T[jcv,icv] 
                    Tf_east   = 0.5 * (T[jcv,icv]+T[jcv,icv+1])  
                    Tf_west   = 0.5 * (T[jcv,icv]+T[jcv,icv-1])   
                    
            elif (jcv==ncv_y-1): # north boundary of the domain
                if ino == 0:  # north east corner 
                    Tf_north  = T[jcv,icv]
                    Tf_south  = 0.5 * (T[jcv,icv]+T[jcv-1,icv])
                    Tf_east   = 0.5 * (T[jcv,icv]+T[jcv,icv+1])
                    Tf_west   = Tleft  
                    
                elif ino == ncv_x-1:  # north east corner 
                    Tf_north  = T[jcv,icv]
                    Tf_south  = 0.5 * (T[jcv,icv]+T[jcv-1,icv])  
                    Tf_east   = Tright
                    Tf_west   = 0.5 * (T[jcv,icv]+T[jcv,icv-1])  
                    
                else: # remainder of north wall  
                    Tf_north  = T[jcv,icv]
                    Tf_south  = 0.5 * (T[jcv,icv]+T[jcv-1,icv])  
                    Tf_east   = 0.5 * (T[jcv,icv]+T[jcv,icv+1])  
                    Tf_west   = 0.5 * (T[jcv,icv]+T[jcv,icv-1])                       
                    
            elif (ino == 0) and (jcv > 0) :  # west border besides corners  
                Tf_north  = 0.5 * (T[jcv,icv]+T[jcv+1,icv])
                Tf_south  = 0.5 * (T[jcv,icv]+T[jcv-1,icv])
                Tf_east   = 0.5 * (T[jcv,icv]+T[jcv,icv+1])   
                Tf_west   = Tleft
                    
            elif (ino == ncv_x-1) and (jcv > 0) :  # east border besides corners  
                Tf_north  = 0.5 * (T[jcv,icv]+T[jcv+1,icv])
                Tf_south  = 0.5 * (T[jcv,icv]+T[jcv-1,icv])
                Tf_east   = Tright
                Tf_west   = 0.5 * (T[jcv,icv]+T[jcv,icv-1])   
                    
            else: # remainder of computational domain 
                Tf_north =  0.5 * (T[jcv,icv]+T[jcv+1,icv])
                Tf_south =  0.5 * (T[jcv,icv]+T[jcv-1,icv])
                Tf_west  =  0.5 * (T[jcv,icv]+T[jcv,icv-1])   
                Tf_east  =  0.5 * (T[jcv,icv]+T[jcv,icv+1])    
                 
            # Convective Part of H        
            F_c = (uf_east*Tf_east - uf_west*Tf_west + vf_north*Tf_north - vf_south*Tf_south)/h
                
            # compute diffusive fluxes
            if (jcv==0):  # south boundary of the domain
                if ino == 0: # south west corner  
                    Tcv_east  = T[jcv,icv+1]
                    Tcv_west  = 2.* Tleft - T[jcv,icv] 
                    Tcv_south = T[jcv,icv] 
                    Tcv_north = T[jcv+1,icv] 
                    
                elif ino == ncv_x-1: # south east corner                    
                    Tcv_east =  2*Tright - T[jcv,icv] 
                    Tcv_west  = T[jcv,icv-1]
                    Tcv_south = T[jcv,icv]
                    Tcv_north = T[jcv+1,icv] 
                    
                else: # remainder or south boundary  
                    Tcv_east  = T[jcv,icv+1]
                    Tcv_west  = T[jcv,icv-1]
                    Tcv_south = T[jcv,icv]    
                    Tcv_north = T[jcv+1,icv]
                        
            elif (jcv==ncv_y-1):                 
                if ino == 0: # north west corner  
                    Tcv_east  = T[jcv,icv+1]
                    Tcv_west  = 2*Tleft - T[jcv,icv]
                    Tcv_south = T[jcv-1,icv] 
                    Tcv_north = T[jcv,icv] 
                    
                elif ino == ncv_x-1: # north east corner  
                    Tcv_east =  2*Tright - T[jcv,icv] 
                    Tcv_west  = T[jcv,icv-1]
                    Tcv_south = T[jcv-1,icv]  
                    Tcv_north = T[jcv,icv]                          
                
                else:# remainder or north boundary  
                    Tcv_east  = T[jcv,icv+1]
                    Tcv_west  = T[jcv,icv-1]
                    Tcv_south = T[jcv-1,icv]
                    Tcv_north = T[jcv,icv]   
             
            elif (ino == 0) and (jcv > 0): # west wall besides cordners 
                Tcv_west  = 2*Tleft - T[jcv,icv]  
                Tcv_east  = T[jcv,icv+1]
                Tcv_south = T[jcv-1,icv]
                Tcv_north = T[jcv+1,icv]
                    
            elif (ino == ncv_x-1) and (jcv > 0) :  # east border besides corners   
                Tcv_west  = T[jcv,icv-1]
                Tcv_east  = 2*Tright - T[jcv,icv] 
                Tcv_south = T[jcv-1,icv]
                Tcv_north = T[jcv+1,icv]
                
            else: #remainder of domain    
                Tcv_west  = T[jcv,icv-1]
                Tcv_east  = T[jcv,icv+1] 
                Tcv_south = T[jcv-1,icv]
                Tcv_north = T[jcv+1,icv]
                
            # Diffusive part of H - second order central difference       
            F_d = (k/rhoref/Cp/(h**2))*(Tcv_east+Tcv_west+Tcv_north+Tcv_south-4.0*T[jcv,ino])
             
            # update the temperature 
            # some error checking here revealed that the temporary temperature was overwriting the temperature before the completion of a full time step
            # pdb.set_trace()
            Tt[jcv,icv] =  T[jcv,icv] + dt * ( -F_c+F_d ) 
    
    # we have now iterated over all x and y and can update the temperature field

    return  

def correct_uv(ncv_x,ncv_y,nno_x,nno_y,h,dt,u,v,ut,vt,p):
    # correct (and update) the velocity for the internal cells
    for jcv in range(ncv_y): 
        for ino in range(1,nno_x-1):
            u[jcv,ino] = ut[jcv,ino] - (dt/h)*(p[jcv,ino]-p[jcv,ino-1])
    for jno in range (1,nno_y-1): 
        for icv in range(ncv_x):
            v[jno,icv] = vt[jno,icv] - (dt/h)*(p[jno,icv]-p[jno-1,icv])
    return

def correct_t(ncv_x,ncv_y,nno_x,nno_y,h,dt,T,Tt):
    # correct (and update) the temperature
    for jcv in range(ncv_y): 
        for icv in range(ncv_x):
            T[jcv,icv] = Tt[jcv,icv]
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
    pl.contourf(X, Y, vmag_plot, alpha=0.8, cmap=cm.jet, levels = 20)
    pl.gca().set_aspect('equal')
    pl.colorbar()
    pl.contour(X, Y, vmag_plot, cmap=cm.jet)
    #pl.quiver(X , Y , u_plot , v_plot )
    pl.xlabel('X')
    pl.ylabel('Y');
    pl.title('U,V');
    
    return 

def plot_t(ncv_x,ncv_y,T): 
            
    x = np.linspace(0.0,1.0,ncv_x)
    y = np.linspace(0.0,1.0,ncv_y) 
    
    # plot vontours of velocity magnitude and velocity vectors
    X, Y = np.meshgrid(x, y)
    fig = pl.figure(figsize=(11, 7), dpi=100)
    pl.contourf(X, Y, T, alpha=0.8, cmap=cm.jet, levels = 50)
    pl.gca().set_aspect('equal')
    pl.colorbar()
    pl.contour(X, Y, T, cmap=cm.jet)
    pl.xlabel('X')
    pl.ylabel('Y')
    pl.title('Temperature');
    
    return

def plot_p(ncv_x,ncv_y,p): 
            
    x = np.linspace(0.0,1.0,ncv_x)
    y = np.linspace(0.0,1.0,ncv_y) 
    
    # plot vontours of velocity magnitude and velocity vectors
    X, Y = np.meshgrid(x, y)
    fig = pl.figure(figsize=(11, 7), dpi=100)
    pl.contourf(X, Y, p, alpha=0.8, cmap=cm.jet, levels = 50)
    pl.gca().set_aspect('equal')
    pl.colorbar()
    pl.contour(X, Y, p, cmap=cm.jet)
    pl.xlabel('X')
    pl.ylabel('Y');
    pl.title('Pressure');
    
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
    nno_x = ncv_x + 1
    nno_y = ncv_y + 1

    # initialize
    u  = np.zeros((ncv_y,nno_x))      # u velocity field
    v  = np.zeros((nno_y,ncv_x))      # v velocity field
    p  = np.zeros((ncv_y,ncv_x))      # pressure   
    T  = np.ones((ncv_y,ncv_x))*Tref  # temperature
    Tt = np.ones((ncv_y,ncv_x))*Tref  # temporary temperature
    ut = np.zeros((ncv_y,nno_x))      # temporary u velocity field
    vt = np.zeros((nno_y,ncv_x))      # temporary v velocity field
     
    # time step and grid size
    h=1.0/ncv_x
    vmax = max(abs(ubc_north),abs(ubc_south),abs(vbc_west),abs(vbc_east)) #If nt walls moving, vmax = 0
    # check stability limit
    dt = deltat              # defaults to 0.02
    dt = min(dt,.2*h*h/mu)   # viscous limit
    
    # if lid driven check peclet number 
    if vmax != 0:
        dt = min(dt,2.0*mu/vmax) # Peclet number limit
    else: 
        for i in range(ncv_y):
            T[i,:] = np.linspace(Tleft,Tright,ncv_x) 
            Tt[i,:] = np.linspace(Tleft,Tright,ncv_x) 

    # grid points (nodes)
    x = np.linspace(0.0,1.0,nno_x)
    y = np.linspace(0.0,1.0,nno_y)

    # initialize the matrix of coefficient
    a_east  = np.zeros((nno_y , nno_x)) 
    a_west  = np.zeros((nno_y , nno_x)) 
    a_north = np.zeros((nno_y , nno_x)) 
    ### pentadiagonal LHS of Poisson solver (holds lap(p) coeffs.)
    a_south = np.zeros((nno_y , nno_x))  
    a_p     = np.zeros((nno_y , nno_x)) 
    b       = np.zeros((nno_y , nno_x)) 
    #will hold div(ut) at each cell center for Poisson solver RHS
    phi     = np.zeros((nno_y , nno_x)) 

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
    tconvvec = []
    timevec  = []
    for istep in range(timesteps):
        #update the temporary velocity field
        solve_uv(ncv_x,ncv_y,nno_x,nno_y,h,dt,ubc_south,ubc_north,vbc_west,vbc_east,u,v,ut,vt,mu,p,beta,Tref,gx,gy,T)
    
        # solve for temperature 
        solve_t(ncv_x,ncv_y,nno_x,nno_y,h,dt,ubc_south,ubc_north,vbc_west,vbc_east,u,v,ut,vt,k,rhoref,Cp,T,Tt,Tleft,Tright,Tref)
        tconv = np.linalg.norm((Tt-T)) # we store a measure of the difference between the temp field at the current and previous time step
        #pdb.set_trace()
        print(tconv)
        tconvvec.append(tconv)
        timevec.append(time+dt)

        # update the temperature field
        # this operation is trivial, but I do it in a separate function to avoid issues related to temp overwriting itself
        correct_t(ncv_x,ncv_y,nno_x,nno_y,h,dt,T,Tt)
        
        #compute the pressure field
        solve_p(ncv_x,ncv_y,nno_x,nno_y,h,dt,p,a_east,a_west,a_north,a_south,a_p,b,ut,vt,"ilu")
        
        #correct the velocity field
        correct_uv(ncv_x,ncv_y,nno_x,nno_y,h,dt,u,v,ut,vt,p)
        
        # advance time
        time = time+dt
        clear()
        print('Timestep ' + str(istep + 1) + ' of ' + str(timesteps))  
    if plot:
        # plot results reporting velocity on all nodes (average)
        plot_uv(ncv_x,ncv_y,nno_x,nno_y,ubc_south,ubc_north,vbc_west,vbc_east,x,y,u,v)
       
        # plot temperature
        plot_t(ncv_x,ncv_y,T) 
        
        # plot pressure
        plot_p(ncv_x,ncv_y,p) 

        # plot some measure to track convergence, in this case l2 norm of the difference between the current temp and temp at previous time step
        fig = pl.figure(figsize=(11, 7), dpi=100)
       # pl.contourf(X, Y, vmag_plot, alpha=0.8, cmap=cm.jet, levels = 20)
        pl.plot(timevec,tconvvec)
        pl.xlabel('time')
        pl.ylabel('Convergence');
        pl.title('l2 T norm');
            
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
