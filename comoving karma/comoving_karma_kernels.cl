// Linear index mapping
inline int get_linear_index(    int2 g,	int2 N  ){
     return g.x + N.x * g.y;
}

// Boundary index mapping
inline int neumannBC(   int2 d, int2 g,	int2 N  ){
     int2 gp = g+d;
     gp.x = gp.x > N.x-1 ? N.x-d.x   : gp.x;
     gp.x = gp.x < 0     ? -1-d.x    : gp.x;
     gp.y = gp.y > N.y-1 ? N.y-d.y   : gp.y;
     gp.y = gp.y < 0     ? -1-d.y    : gp.y; 
     return get_linear_index(gp, N);			
}
inline int periodicBC(  int2 d,	int2 g,	int2 N  ){
     int2 gp = g+d;
     gp.x = gp.x > N.x-1 ? gp.x-N.x : gp.x;
     gp.x = gp.x < 0     ? N.x+gp.x : gp.x;
     gp.y = gp.y > N.y-1 ? gp.y-N.y : gp.y;
     gp.y = gp.y < 0     ? N.y+gp.y : gp.y; 
     return get_linear_index(gp, N);	
}

// sigmoid functions
inline real H(  real x, real k  ){
     return 0.5 * (1.0 + tanh(k * x));  
}
inline real dH( real x, real k  ){
     real sechkx = 1.0/cosh(k * x);
     return 0.5 * k * sechkx * sechkx;
}

// interpolation functions {O(h^1), O(h^3), O(h^4)}
inline real2 hermite1(real2 x0, real2 x1, real2 m0, real2 m1, real t){
     real2 p   = x0 + (x1 - x0) * t;                       // + O(h^2)
     return p;
}
inline real2 hermite2(real2 x0, real2 x1, real2 m0, real2 m1, real t){
//     real2 p   = x0 + (x1 - x0) * t + t * (1.0 - t) * (m1 - m0);                       // + O(h^2)
    real2 p   = (t-1.0)*(t-1.0)*(2.0*t + 1.0) * x0 
               + t*(t-1.0)*(t-1.0) * m0
               + t*t*(3.0-2.0*t) * x1 
               + t*t*(t-1.0) * m1;                         // + O(h^4)
     return p;
}
inline real2 hermite3(real2 x0, real2 x1, real2 m0, real2 m1, real t){
     real2 p   = (t-1.0)*(t-1.0)*(2.0*t + 1.0) * x0 
               + t*(t-1.0)*(t-1.0) * m0
               + t*t*(3.0-2.0*t) * x1 
               + t*t*(t-1.0) * m1;                         // + O(h^4)
     return p;
}
inline real2 hermite4(real2 x0, real2 x1, real2 m0, real2 m1, real2 m2, real t){ // eta == 1/3
     real2 p   = ((1.0 + 2.0*t + 9.0*t*t)*(t-1.0)*(t-1.0)) * x0 
               + (t*(t-1.0)*(t-1.0)) * m0
               + (-t*t*(6.0 - 16.0*t + 9.0*t*t)) * x1 
               + ((t-1.0)*(9.0*t-5.0)*t*t/4.0) * m1
               + (27.0*t*t*(t-1.0)*(t-1.0)/4.0) * m2;      // + O(h^5)
     return p; 
}
// Vector function for evolution F: R^2 → R^2
inline real2 F(real u, real v, real beta, real u_star, real k, 
     real eps_u, real eps_v, real D_u, real D_v){
     real f 	= pown(u,2);
     f	= -u + f*(u_star - pown(v,4)) * H(3.0-u,1.0);
     real g 	= v-1.0;
     g	= -v + beta * H(u-1.0,k) + g*H(g,k);
     return	(real2)(eps_u*f,eps_v*g);
}
// Matrix function for tangent evolution, dF: R^2 → R^4
inline real4 dF(real u, real v, real beta, real u_star, real k, 
     real eps_u, real eps_v, real D_u, real D_v){
     real u2     = u*u; real v3 = pown(v,3); real v4 = v*v3;
     real Auu    = -1.0 + u*1.5415 - u*v4 - u*(1.5415 - v4)*tanh(u - 3.0) 
     - (u2*(1.5415 - v4)*pown(1.0/cosh(u - 3.0),2))*0.5;
     real Auv    = 2.0*u2*v3*(tanh(u - 3.0) - 1.0);
     real Avu    = beta * dH(u-1.0, k);
     real Avv    = H(v-1.0, k) + (v-1.0)*dH(v-1.0, k) - 1.0;
     return	 (real4)(eps_u*Auu, eps_u*Auv, eps_v*Avu, eps_v*Avv);
}
// (sol,fwd/adj) -> (d(sol)/dt,d(fwd/adj)/dt)
__kernel void neumann_dxdt(
     __global real *u,   __global real *v,
     __global real *up,  __global real *vp,
     __global real *ku,  __global real *kv,
     __global real *kup, __global real *kvp, 
     real beta, real u_star, real k, 
     real eps_u, real eps_v, real D_u, real D_v,
     real GAMMA, real core_x, real core_y, real c_x, real c_y, real omega, 
     real fwd_adj){
// Computes the rhs of d(u,v,up,vp)/dt and stores the components in the buffers
// (ku,kv,kup,kvp), respectively. 
// First computing D_ij*Laplacian[.], then c[omega].grad(.), 
// and then finally (f,df).

// The Laplacian stencil is:
//              [(1-g)/2     g     (1-g)/2]
//              [g        -2(1+g)        g]             + O(|h|^4)
//              [(1-g)/2     g     (1-g)/2]
// The gradient stencil is:
//      [1/12     -2/3       0       2/3     -1/12]     + O(|h|^4)
// Combined the stencil is?

     int2 g    = (int2)(get_global_id(0),	get_global_id(1));
     int2 N    = (int2)(get_global_size(0),get_global_size(1));

     int gid   = get_linear_index(g, N);

     real2 x   = (real2)( u[gid], 	 v[gid]);
     real2 xp	= (real2)(up[gid], 	vp[gid]);

     real2 kx  = -2.0*(1.0 + GAMMA) * x;
     real2 kxp = -2.0*(1.0 + GAMMA) * xp;

     real2 grad_u, grad_v, grad_up, grad_vp, c;

// Apply L[.] to (u,v,up,vp) in turn: 
     kx.s0     += ( u[neumannBC((int2)(+1,+0), g, N)] 
               +    u[neumannBC((int2)(-1,+0), g, N)] ) * GAMMA
               +  ( u[neumannBC((int2)(+0,+1), g, N)] 
               +    u[neumannBC((int2)(+0,-1), g, N)] ) * GAMMA;
     kx.s0     += ( u[neumannBC((int2)(+1,-1), g, N)] 
               +    u[neumannBC((int2)(-1,-1), g, N)] ) * (1.0 - GAMMA)*0.5
               +  ( u[neumannBC((int2)(-1,+1), g, N)] 
               +    u[neumannBC((int2)(+1,+1), g, N)] ) * (1.0 - GAMMA)*0.5;
     kx.s0     *= D_u;    

     kx.s1     += ( v[neumannBC((int2)(+1,+0), g, N)] 
               +    v[neumannBC((int2)(-1,+0), g, N)] ) * GAMMA
               +  ( v[neumannBC((int2)(+0,+1), g, N)] 
               +    v[neumannBC((int2)(+0,-1), g, N)] ) * GAMMA;
     kx.s1     += ( v[neumannBC((int2)(+1,-1), g, N)] 
               +    v[neumannBC((int2)(-1,-1), g, N)] ) * (1.0 - GAMMA)*0.5
               +  ( v[neumannBC((int2)(-1,+1), g, N)] 
               +    v[neumannBC((int2)(+1,+1), g, N)] ) * (1.0 - GAMMA)*0.5;
     kx.s1     *= D_v;

     kxp.s0    += ( up[neumannBC((int2)(+1,+0), g, N)] 
               +    up[neumannBC((int2)(-1,+0), g, N)] ) * GAMMA
               +  ( up[neumannBC((int2)(+0,+1), g, N)] 
               +    up[neumannBC((int2)(+0,-1), g, N)] ) * GAMMA;
     kxp.s0    += ( up[neumannBC((int2)(+1,-1), g, N)] 
               +    up[neumannBC((int2)(-1,-1), g, N)] ) * (1.0 - GAMMA)*0.5
               +  ( up[neumannBC((int2)(-1,+1), g, N)] 
               +    up[neumannBC((int2)(+1,+1), g, N)] ) * (1.0 - GAMMA)*0.5;
     kxp.s0    *= D_u;

     kxp.s1    += ( vp[neumannBC((int2)(+1,+0), g, N)] 
               +    vp[neumannBC((int2)(-1,+0), g, N)] ) * GAMMA
               +  ( vp[neumannBC((int2)(+0,+1), g, N)] 
               +    vp[neumannBC((int2)(+0,-1), g, N)] ) * GAMMA;
     kxp.s1    += ( vp[neumannBC((int2)(+1,-1), g, N)] 
               +    vp[neumannBC((int2)(-1,-1), g, N)] ) * (1.0 - GAMMA)*0.5
               +  ( vp[neumannBC((int2)(-1,+1), g, N)] 
               +    vp[neumannBC((int2)(+1,+1), g, N)] ) * (1.0 - GAMMA)*0.5;
     kxp.s1    *= D_v;

     grad_u    = (real2)(u[neumannBC((int2)(-2,+0), g, N)]
               -         u[neumannBC((int2)(+2,+0), g, N)],
                         u[neumannBC((int2)(+0,-2), g, N)]
               -         u[neumannBC((int2)(+0,+2), g, N)]);
     grad_u    +=(real2)(u[neumannBC((int2)(+1,+0), g, N)]
               -         u[neumannBC((int2)(-1,+0), g, N)],
                         u[neumannBC((int2)(+0,+1), g, N)]
               -         u[neumannBC((int2)(+0,-1), g, N)])*8.0;
     grad_u    /= 12.0;    

     grad_v    = (real2)(v[neumannBC((int2)(-2,+0), g, N)]
               -         v[neumannBC((int2)(+2,+0), g, N)],
                         v[neumannBC((int2)(+0,-2), g, N)]
               -         v[neumannBC((int2)(+0,+2), g, N)]);
     grad_v    +=(real2)(v[neumannBC((int2)(+1,+0), g, N)]
               -         v[neumannBC((int2)(-1,+0), g, N)],
                         v[neumannBC((int2)(+0,+1), g, N)]
               -         v[neumannBC((int2)(+0,-1), g, N)])*8.0;
     grad_v    /= 12.0;

     grad_up   = (real2)(up[neumannBC((int2)(-2,+0), g, N)]
               -         up[neumannBC((int2)(+2,+0), g, N)],
                         up[neumannBC((int2)(+0,-2), g, N)]
               -         up[neumannBC((int2)(+0,+2), g, N)]);
     grad_up   +=(real2)(up[neumannBC((int2)(+1,+0), g, N)]
               -         up[neumannBC((int2)(-1,+0), g, N)],
                         up[neumannBC((int2)(+0,+1), g, N)]
                    -    up[neumannBC((int2)(+0,-1), g, N)])*8.0;
     grad_up   /= 12.0;    

     grad_vp   = (real2)(vp[neumannBC((int2)(-2,+0), g, N)]
               -         vp[neumannBC((int2)(+2,+0), g, N)],
                         vp[neumannBC((int2)(+0,-2), g, N)]
               -         vp[neumannBC((int2)(+0,+2), g, N)]);
     grad_vp   +=(real2)(vp[neumannBC((int2)(+1,+0), g, N)]
               -         vp[neumannBC((int2)(-1,+0), g, N)],
                         vp[neumannBC((int2)(+0,+1), g, N)]
               -         vp[neumannBC((int2)(+0,-1), g, N)])*8.0;
     grad_vp   /= 12.0;

     c         = (real2)(c_x - omega*(g.y-core_y), c_y + omega*(g.x-core_x));

     kx        += (real2)(dot(c,grad_u), dot(c,grad_v));
     kxp       += sign(fwd_adj)*(real2)(dot(c,grad_up),dot(c,grad_vp));

     real4 DF  = dF(x.s0, x.s1, beta, u_star, k, eps_u, eps_v, D_u, D_v);

// int fwd_adj \in {-1,1} for adjoint, forward, respectively.
     if( fwd_adj < 0.0 ){ // fwd_adj = sign(dt/dt'), t' a directed parametrization of solution
          DF   = DF.s0213;  // if backwards time, twiddle
     }

// Add N( x_n ) to existing tangents, yielding kx_n.
     kx        += F(x.s0, x.s1, beta, u_star, k, eps_u, eps_v, D_u, D_v);
     kxp       += (real2)(DF.s0 * xp.s0 + DF.s1 * xp.s1, DF.s2 * xp.s0 + DF.s3 * xp.s1);

// Bookkeeping.
     ku[gid]   =	kx.s0;
     kv[gid] 	=	kx.s1;
     kup[gid]	=	kxp.s0;
     kvp[gid]	=	kxp.s1;
}
__kernel void periodic_dxdt(
     __global real *u,   __global real *v,
     __global real *up,  __global real *vp,
     __global real *ku,  __global real *kv,
     __global real *kup, __global real *kvp, 
     real beta, real u_star, real k, 
     real eps_u, real eps_v, real D_u, real D_v,
     real GAMMA, real core_x, real core_y, real c_x, real c_y, real omega, 
     real fwd_adj){
// Computes the rhs of d(u,v,up,vp)/dt and stores the components in the buffers
// (ku,kv,kup,kvp), respectively. 
// First computing D_ij*Laplacian[.], then c[omega].grad(.), 
// and then finally (f,df).

// The Laplacian stencil is:
//              [(1-g)/2     g     (1-g)/2]
//              [g        -2(1+g)        g]             + O(|h|^4)
//              [(1-g)/2     g     (1-g)/2]
// The gradient stencil is:
//      [1/12     -2/3       0       2/3     -1/12]     + O(|h|^4)
// Combined the stencil is?

     int2 g    = (int2)(get_global_id(0),	get_global_id(1));
     int2 N    = (int2)(get_global_size(0),get_global_size(1));

     int gid   = get_linear_index(g, N);

     real2 x   = (real2)( u[gid], 	 v[gid]);
     real2 xp  = (real2)(up[gid], 	vp[gid]);

     real2 kx  = -2.0*(1.0 + GAMMA) * x;
     real2 kxp = -2.0*(1.0 + GAMMA) * xp;

     real2 grad_u, grad_v, grad_up, grad_vp, c;

// Apply L[.] to (u,v,up,vp) in turn: 
     kx.s0     += ( u[periodicBC((int2)(+1,+0), g, N)] 
               +    u[periodicBC((int2)(-1,+0), g, N)] ) * GAMMA
               +  ( u[periodicBC((int2)(+0,+1), g, N)] 
               +    u[periodicBC((int2)(+0,-1), g, N)] ) * GAMMA;
     kx.s0     += ( u[periodicBC((int2)(+1,-1), g, N)] 
               +    u[periodicBC((int2)(-1,-1), g, N)] ) * (1.0 - GAMMA)*0.5
               +  ( u[periodicBC((int2)(-1,+1), g, N)] 
               +    u[periodicBC((int2)(+1,+1), g, N)] ) * (1.0 - GAMMA)*0.5;
     kx.s0     *= D_u;    

     kx.s1     += ( v[periodicBC((int2)(+1,+0), g, N)] 
               +    v[periodicBC((int2)(-1,+0), g, N)] ) * GAMMA
               +  ( v[periodicBC((int2)(+0,+1), g, N)] 
               +    v[periodicBC((int2)(+0,-1), g, N)] ) * GAMMA;
     kx.s1     += ( v[periodicBC((int2)(+1,-1), g, N)] 
               +    v[periodicBC((int2)(-1,-1), g, N)] ) * (1.0 - GAMMA)*0.5
               +  ( v[periodicBC((int2)(-1,+1), g, N)] 
               +    v[periodicBC((int2)(+1,+1), g, N)] ) * (1.0 - GAMMA)*0.5;
     kx.s1     *= D_v;

     kxp.s0    += ( up[periodicBC((int2)(+1,+0), g, N)] 
               +    up[periodicBC((int2)(-1,+0), g, N)] ) * GAMMA
               +  ( up[periodicBC((int2)(+0,+1), g, N)] 
               +    up[periodicBC((int2)(+0,-1), g, N)] ) * GAMMA;
     kxp.s0    += ( up[periodicBC((int2)(+1,-1), g, N)] 
               +    up[periodicBC((int2)(-1,-1), g, N)] ) * (1.0 - GAMMA)*0.5
               +  ( up[periodicBC((int2)(-1,+1), g, N)] 
               +    up[periodicBC((int2)(+1,+1), g, N)] ) * (1.0 - GAMMA)*0.5;
     kxp.s0    *= D_u;

     kxp.s1    += ( vp[periodicBC((int2)(+1,+0), g, N)] 
               +    vp[periodicBC((int2)(-1,+0), g, N)] ) * GAMMA
               +  ( vp[periodicBC((int2)(+0,+1), g, N)] 
               +    vp[periodicBC((int2)(+0,-1), g, N)] ) * GAMMA;
     kxp.s1    += ( vp[periodicBC((int2)(+1,-1), g, N)] 
               +    vp[periodicBC((int2)(-1,-1), g, N)] ) * (1.0 - GAMMA)*0.5
               +  ( vp[periodicBC((int2)(-1,+1), g, N)] 
               +    vp[periodicBC((int2)(+1,+1), g, N)] ) * (1.0 - GAMMA)*0.5;
     kxp.s1    *= D_v;

     grad_u    = (real2)(u[periodicBC((int2)(-2,+0), g, N)]
               -         u[periodicBC((int2)(+2,+0), g, N)],
                         u[periodicBC((int2)(+0,-2), g, N)]
               -         u[periodicBC((int2)(+0,+2), g, N)]);
     grad_u    +=(real2)(u[periodicBC((int2)(+1,+0), g, N)]
               -         u[periodicBC((int2)(-1,+0), g, N)],
                         u[periodicBC((int2)(+0,+1), g, N)]
               -         u[periodicBC((int2)(+0,-1), g, N)])*8.0;
     grad_u    /= 12.0;    

     grad_v    = (real2)(v[periodicBC((int2)(-2,+0), g, N)]
               -         v[periodicBC((int2)(+2,+0), g, N)],
                         v[periodicBC((int2)(+0,-2), g, N)]
               -         v[periodicBC((int2)(+0,+2), g, N)]);
     grad_v    +=(real2)(v[periodicBC((int2)(+1,+0), g, N)]
               -         v[periodicBC((int2)(-1,+0), g, N)],
                         v[periodicBC((int2)(+0,+1), g, N)]
               -         v[periodicBC((int2)(+0,-1), g, N)])*8.0;
     grad_v    /= 12.0;

     grad_up   = (real2)(up[periodicBC((int2)(-2,+0), g, N)]
               -         up[periodicBC((int2)(+2,+0), g, N)],
                         up[periodicBC((int2)(+0,-2), g, N)]
               -         up[periodicBC((int2)(+0,+2), g, N)]);
     grad_up   +=(real2)(up[periodicBC((int2)(+1,+0), g, N)]
               -         up[periodicBC((int2)(-1,+0), g, N)],
                         up[periodicBC((int2)(+0,+1), g, N)]
               -         up[periodicBC((int2)(+0,-1), g, N)])*8.0;
     grad_up   /= 12.0;    

     grad_vp   = (real2)(vp[periodicBC((int2)(-2,+0), g, N)]
               -         vp[periodicBC((int2)(+2,+0), g, N)],
                         vp[periodicBC((int2)(+0,-2), g, N)]
               -         vp[periodicBC((int2)(+0,+2), g, N)]);
     grad_vp   +=(real2)(vp[periodicBC((int2)(+1,+0), g, N)]
               -         vp[periodicBC((int2)(-1,+0), g, N)],
                         vp[periodicBC((int2)(+0,+1), g, N)]
               -         vp[periodicBC((int2)(+0,-1), g, N)])*8.0;
     grad_vp   /= 12.0;

     c         = (real2)(c_x - omega*(g.y-core_y), c_y + omega*(g.x-core_x));

     kx        += (real2)(dot(c,grad_u), dot(c,grad_v));
     kxp       += sign(fwd_adj)*(real2)(dot(c,grad_up),dot(c,grad_vp));

     real4 DF  = dF(x.s0, x.s1, beta, u_star, k, eps_u, eps_v, D_u, D_v);

// int fwd_adj \in {-1,1} for adjoint, forward, respectively.
     if( fwd_adj < 0.0 ){ // fwd_adj = sign(dt/dt'), t' a directed parametrization of solution
          DF   = DF.s0213;  // if backwards time, twiddle
     }

// Add N( x_n ) to existing tangents, yielding kx_n.
     kx 	     += F(x.s0, x.s1, beta, u_star, k, eps_u, eps_v, D_u, D_v);
     kxp       += (real2)(DF.s0 * xp.s0 + DF.s1 * xp.s1, DF.s2 * xp.s0 + DF.s3 * xp.s1);

// Bookkeeping.
     ku[gid]   =	kx.s0;
     kv[gid] 	=	kx.s1;
     kup[gid]	=	kxp.s0;
     kvp[gid]	=	kxp.s1;
}
__kernel void initial_dxdt(
    __global real *u,   __global real *v,
    __global real *up,  __global real *vp,
    __global real *ku,  __global real *kv,
    __global real *kup, __global real *kvp, 
    real beta, real u_star, real k, 
    real eps_u, real eps_v, real D_u, real D_v,
    real GAMMA, real core_x, real core_y, real c_x, real c_y, real omega, 
    real fwd_adj){
// Computes the rhs of d(u,v,up,vp)/dt and stores the components in the buffers
// (ku,kv,kup,kvp), respectively. 
// First computing D_ij*Laplacian[.], then c[omega].grad(.), 
// and then finally (f,df).

// The Laplacian stencil is:
//              [(1-g)/2     g     (1-g)/2]
//              [g        -2(1+g)        g]             + O(|h|^4)
//              [(1-g)/2     g     (1-g)/2]
// The gradient stencil is:
//      [1/12     -2/3       0       2/3     -1/12]     + O(|h|^4)
// Combined the stencil is?

    int2 g    = (int2)(get_global_id(0),	get_global_id(1));
    int2 N    = (int2)(get_global_size(0),get_global_size(1));

    if((g.x!=0)&&(g.x!=N.x-1)&&(g.y!=0)&&(g.y!=N.y-1)){
        int gid   = get_linear_index(g, N);
        real2 x   = (real2)( u[gid], 	 v[gid]);
        real2 xp  = (real2)(up[gid], 	vp[gid]);

        real2 kx  = -2.0*(1.0 + GAMMA) * x;
        real2 kxp = -2.0*(1.0 + GAMMA) * xp;

        real2 grad_u, grad_v, grad_up, grad_vp, c;
        int8 indices = (int8)(  get_linear_index(g+(int2)(+1,+0),N),
                                get_linear_index(g+(int2)(-1,+0),N),
                                get_linear_index(g+(int2)(+0,+1),N),
                                get_linear_index(g+(int2)(+0,-1),N),
                                get_linear_index(g+(int2)(+1,-1),N),
                                get_linear_index(g+(int2)(-1,-1),N),
                                get_linear_index(g+(int2)(-1,+1),N),
                                get_linear_index(g+(int2)(+1,+1),N) );

        // Apply L[.] to (u,v,up,vp) in turn: 
        kx.s0   += ( u[indices.s0] 
                +    u[indices.s1] ) * GAMMA
                +  ( u[indices.s2] 
                +    u[indices.s3] ) * GAMMA;
        kx.s0   += ( u[indices.s4] 
                +    u[indices.s5] ) * (1.0 - GAMMA)*0.5
                +  ( u[indices.s6] 
                +    u[indices.s7] ) * (1.0 - GAMMA)*0.5;
        kx.s0   *= D_u;    

        kx.s1   += ( v[indices.s0] 
                +    v[indices.s1] ) * GAMMA
                +  ( v[indices.s2] 
                +    v[indices.s3] ) * GAMMA;
        kx.s1   += ( v[indices.s4] 
                +    v[indices.s5] ) * (1.0 - GAMMA)*0.5
                +  ( v[indices.s6] 
                +    v[indices.s7] ) * (1.0 - GAMMA)*0.5;
        kx.s1   *= D_v;

        kxp.s0  += ( up[indices.s0] 
                +    up[indices.s1] ) * GAMMA
                +  ( up[indices.s2] 
                +    up[indices.s3] ) * GAMMA;
        kxp.s0  += ( up[indices.s4] 
                +    up[indices.s5] ) * (1.0 - GAMMA)*0.5
                +  ( up[indices.s6] 
                +    up[indices.s7] ) * (1.0 - GAMMA)*0.5;
        kxp.s0  *= D_u;

        kxp.s1  += ( vp[indices.s0] 
                +    vp[indices.s1] ) * GAMMA
                +  ( vp[indices.s2] 
                +    vp[indices.s3] ) * GAMMA;
        kxp.s1  += ( vp[indices.s4] 
                +    vp[indices.s5] ) * (1.0 - GAMMA)*0.5
                +  ( vp[indices.s6] 
                +    vp[indices.s7] ) * (1.0 - GAMMA)*0.5;
        kxp.s1  *= D_v;
   
        indices = (int8)(  neumannBC((int2)(-2,+0), g, N),
                           neumannBC((int2)(+2,+0), g, N),
                           neumannBC((int2)(+0,-2), g, N),
                           neumannBC((int2)(+0,+2), g, N),
                           get_linear_index(g+(int2)(+1,+0),N),
                           get_linear_index(g+(int2)(-1,+0),N),
                           get_linear_index(g+(int2)(+0,+1),N),
                           get_linear_index(g+(int2)(+0,-1),N) );

        grad_u  = (real2)(u[indices.s0]
                -         u[indices.s1],
                          u[indices.s2]
                -         u[indices.s3]);
        grad_u  +=(real2)(u[indices.s4]
                -         u[indices.s5],
                          u[indices.s6]
                -         u[indices.s7])*8.0;
        grad_u  /= 12.0;    

        grad_v  = (real2)(v[indices.s0]
                -         v[indices.s1],
                          v[indices.s2]
                -         v[indices.s3]);
        grad_v  +=(real2)(v[indices.s4]
                -         v[indices.s5],
                          v[indices.s6]
                -         v[indices.s7])*8.0;
        grad_v  /= 12.0;

        grad_up = (real2)(up[indices.s0]
                -         up[indices.s1],
                          up[indices.s2]
                -         up[indices.s3]);
        grad_up +=(real2)(up[indices.s4]
                -         up[indices.s5],
                          up[indices.s6]
                -         up[indices.s7])*8.0;
        grad_up /= 12.0;    

        grad_vp = (real2)(vp[indices.s0]
                -         vp[indices.s1],
                          vp[indices.s2]
                -         vp[indices.s3]);
        grad_vp +=(real2)(vp[indices.s4]
                -         vp[indices.s5],
                          vp[indices.s6]
                -         vp[indices.s7])*8.0;
        grad_vp /= 12.0;

        c       = (real2)(c_x - omega*(g.y-core_y), c_y + omega*(g.x-core_x));

        kx      += (real2)(dot(c,grad_u), dot(c,grad_v));
        kxp     += sign(fwd_adj)*(real2)(dot(c,grad_up),dot(c,grad_vp));

        real4 DF  = dF(x.s0, x.s1, beta, u_star, k, eps_u, eps_v, D_u, D_v);

        // int fwd_adj \in {-1,1} for adjoint, forward, respectively.
        if( fwd_adj < 0.0 ){ // fwd_adj = sign(dt/dt'), t' a directed parametrization of solution
            DF  = DF.s0213;  // if backwards time, twiddle
        }

        // Add N( x_n ) to existing tangents, yielding kx_n.
        kx      += F(x.s0, x.s1, beta, u_star, k, eps_u, eps_v, D_u, D_v);
        kxp     += (real2)(DF.s0 * xp.s0 + DF.s1 * xp.s1, DF.s2 * xp.s0 + DF.s3 * xp.s1);

        // Bookkeeping.
        ku[gid] =	kx.s0;
        kv[gid] =	kx.s1;
        kup[gid]    =	kxp.s0;
        kvp[gid]	=	kxp.s1;
    }
}
__kernel void circle_dxdt(
    __global real *u,   __global real *v,
    __global real *up,  __global real *vp,
    __global real *ku,  __global real *kv,
    __global real *kup, __global real *kvp, 
    real beta, real u_star, real k, 
    real eps_u, real eps_v, real D_u, real D_v,
    real GAMMA, real origin_x, real origin_y, real c_x, real c_y, real omega, 
    real fwd_adj){
// Computes the rhs of d(u,v,up,vp)/dt and stores the components in the buffers
// (ku,kv,kup,kvp), respectively. 
// First computing D_ij*Laplacian[.], then c[omega].grad(.), 
// and then finally (f,df).

// The Laplacian stencil is:
//              [(1-g)/2     g     (1-g)/2]
//              [g        -2(1+g)        g]             + O(|h|^4)
//              [(1-g)/2     g     (1-g)/2]
// The gradient stencil is:
//      [1/12     -2/3       0       2/3     -1/12]     + O(|h|^4)
// Combined the stencil is?

    int2 g    = (int2)(get_global_id(0),	get_global_id(1));
    int2 N    = (int2)(get_global_size(0),get_global_size(1));
    real r    = length((real2)(g.x,g.y)-(real2)(origin_x,origin_y));
    
    real R    = N.x * 0.5 * 0.75;
    R = fmin(R, fmin(origin_x, N.x-origin_x)); // R < {l1, N1-l1}
    R = fmin(R, fmin(origin_y, N.y-origin_y)); // R < {l2, N2-l2}
     
    if(r < R - 1.0){
        int gid   = get_linear_index(g, N);
        real2 x   = (real2)( u[gid], 	 v[gid]);
        real2 xp  = (real2)(up[gid], 	vp[gid]);

        real2 kx  = -2.0*(1.0 + GAMMA) * x;
        real2 kxp = -2.0*(1.0 + GAMMA) * xp;

        real2 grad_u, grad_v, grad_up, grad_vp, c;
        int8 indices = (int8)(  get_linear_index(g+(int2)(+1,+0),N),
                                get_linear_index(g+(int2)(-1,+0),N),
                                get_linear_index(g+(int2)(+0,+1),N),
                                get_linear_index(g+(int2)(+0,-1),N),
                                get_linear_index(g+(int2)(+1,-1),N),
                                get_linear_index(g+(int2)(-1,-1),N),
                                get_linear_index(g+(int2)(-1,+1),N),
                                get_linear_index(g+(int2)(+1,+1),N) );

        // Apply L[.] to (u,v,up,vp) in turn: 
        kx.s0   += ( u[indices.s0] 
                +    u[indices.s1] ) * GAMMA
                +  ( u[indices.s2] 
                +    u[indices.s3] ) * GAMMA;
        kx.s0   += ( u[indices.s4] 
                +    u[indices.s5] ) * (1.0 - GAMMA)*0.5
                +  ( u[indices.s6] 
                +    u[indices.s7] ) * (1.0 - GAMMA)*0.5;
        kx.s0   *= D_u;    

        kx.s1   += ( v[indices.s0] 
                +    v[indices.s1] ) * GAMMA
                +  ( v[indices.s2] 
                +    v[indices.s3] ) * GAMMA;
        kx.s1   += ( v[indices.s4] 
                +    v[indices.s5] ) * (1.0 - GAMMA)*0.5
                +  ( v[indices.s6] 
                +    v[indices.s7] ) * (1.0 - GAMMA)*0.5;
        kx.s1   *= D_v;

        kxp.s0  += ( up[indices.s0] 
                +    up[indices.s1] ) * GAMMA
                +  ( up[indices.s2] 
                +    up[indices.s3] ) * GAMMA;
        kxp.s0  += ( up[indices.s4] 
                +    up[indices.s5] ) * (1.0 - GAMMA)*0.5
                +  ( up[indices.s6] 
                +    up[indices.s7] ) * (1.0 - GAMMA)*0.5;
        kxp.s0  *= D_u;

        kxp.s1  += ( vp[indices.s0] 
                +    vp[indices.s1] ) * GAMMA
                +  ( vp[indices.s2] 
                +    vp[indices.s3] ) * GAMMA;
        kxp.s1  += ( vp[indices.s4] 
                +    vp[indices.s5] ) * (1.0 - GAMMA)*0.5
                +  ( vp[indices.s6] 
                +    vp[indices.s7] ) * (1.0 - GAMMA)*0.5;
        kxp.s1  *= D_v;
   
        indices = (int8)(  neumannBC((int2)(-2,+0), g, N),
                           neumannBC((int2)(+2,+0), g, N),
                           neumannBC((int2)(+0,-2), g, N),
                           neumannBC((int2)(+0,+2), g, N),
                           get_linear_index(g+(int2)(+1,+0),N),
                           get_linear_index(g+(int2)(-1,+0),N),
                           get_linear_index(g+(int2)(+0,+1),N),
                           get_linear_index(g+(int2)(+0,-1),N) );

        grad_u  = (real2)(u[indices.s0]
                -         u[indices.s1],
                          u[indices.s2]
                -         u[indices.s3]);
        grad_u  +=(real2)(u[indices.s4]
                -         u[indices.s5],
                          u[indices.s6]
                -         u[indices.s7])*8.0;
        grad_u  /= 12.0;    

        grad_v  = (real2)(v[indices.s0]
                -         v[indices.s1],
                          v[indices.s2]
                -         v[indices.s3]);
        grad_v  +=(real2)(v[indices.s4]
                -         v[indices.s5],
                          v[indices.s6]
                -         v[indices.s7])*8.0;
        grad_v  /= 12.0;

        grad_up = (real2)(up[indices.s0]
                -         up[indices.s1],
                          up[indices.s2]
                -         up[indices.s3]);
        grad_up +=(real2)(up[indices.s4]
                -         up[indices.s5],
                          up[indices.s6]
                -         up[indices.s7])*8.0;
        grad_up /= 12.0;    

        grad_vp = (real2)(vp[indices.s0]
                -         vp[indices.s1],
                          vp[indices.s2]
                -         vp[indices.s3]);
        grad_vp +=(real2)(vp[indices.s4]
                -         vp[indices.s5],
                          vp[indices.s6]
                -         vp[indices.s7])*8.0;
        grad_vp /= 12.0;

        c       = (real2)(c_x - omega*(g.y-origin_y), c_y + omega*(g.x-origin_x));

        kx      += (real2)(dot(c,grad_u), dot(c,grad_v));
        kxp     += sign(fwd_adj)*(real2)(dot(c,grad_up),dot(c,grad_vp));

        real4 DF  = dF(x.s0, x.s1, beta, u_star, k, eps_u, eps_v, D_u, D_v);

        // int fwd_adj \in {-1,1} for adjoint, forward, respectively.
        if( fwd_adj < 0.0 ){ // fwd_adj = sign(dt/dt'), t' a directed parametrization of solution
            DF  = DF.s0213;  // if backwards time, twiddle
        }

        // Add N( x_n ) to existing tangents, yielding kx_n.
        kx      += F(x.s0, x.s1, beta, u_star, k, eps_u, eps_v, D_u, D_v);
        kxp     += (real2)(DF.s0 * xp.s0 + DF.s1 * xp.s1, DF.s2 * xp.s0 + DF.s3 * xp.s1);

        // Bookkeeping.
        ku[gid] =	kx.s0;
        kv[gid] =	kx.s1;
        kup[gid]    =	kxp.s0;
        kvp[gid]	=	kxp.s1;
    }
}
// Solution state (u,v) specific
__kernel void sol_circle_dxdt(
    __global real *u,   __global real *v,
    __global real *ku,  __global real *kv,
    real beta, real u_star, real k, 
    real eps_u, real eps_v, real D_u, real D_v,
    real GAMMA, real origin_x, real origin_y, real c_x, real c_y, real omega, 
    real fwd_adj){
// Computes the rhs of d(u,v,up,vp)/dt and stores the components in the buffers
// (ku,kv,kup,kvp), respectively. 
// First computing D_ij*Laplacian[.], then c[omega].grad(.), 
// and then finally (f,df).

// The Laplacian stencil is:
//              [(1-g)/2     g     (1-g)/2]
//              [g        -2(1+g)        g]             + O(|h|^4)
//              [(1-g)/2     g     (1-g)/2]
// The gradient stencil is:
//      [1/12     -2/3       0       2/3     -1/12]     + O(|h|^4)
// Combined the stencil is?

    int2 g    = (int2)(get_global_id(0),	get_global_id(1));
    int2 N    = (int2)(get_global_size(0),get_global_size(1));
    real r    = length((real2)(g.x,g.y)-(real2)(origin_x,origin_y));
    
    real R    = N.x * 0.5 * 0.75;
    R = fmin(R, fmin(origin_x, N.x-origin_x)); // R < {l1, N1-l1}
    R = fmin(R, fmin(origin_y, N.y-origin_y)); // R < {l2, N2-l2}
     
    if(r < R - 1.0){
        int gid   = get_linear_index(g, N);
        real2 x   = (real2)( u[gid], 	 v[gid]);

        real2 kx  = -2.0*(1.0 + GAMMA) * x;

        real2 grad_u, grad_v, c;
        int8 indices = (int8)(  get_linear_index(g+(int2)(+1,+0),N),
                                get_linear_index(g+(int2)(-1,+0),N),
                                get_linear_index(g+(int2)(+0,+1),N),
                                get_linear_index(g+(int2)(+0,-1),N),
                                get_linear_index(g+(int2)(+1,-1),N),
                                get_linear_index(g+(int2)(-1,-1),N),
                                get_linear_index(g+(int2)(-1,+1),N),
                                get_linear_index(g+(int2)(+1,+1),N) );

        // Apply L[.] to (u,v,up,vp) in turn: 
        kx.s0   += ( u[indices.s0] 
                +    u[indices.s1] ) * GAMMA
                +  ( u[indices.s2] 
                +    u[indices.s3] ) * GAMMA;
        kx.s0   += ( u[indices.s4] 
                +    u[indices.s5] ) * (1.0 - GAMMA)*0.5
                +  ( u[indices.s6] 
                +    u[indices.s7] ) * (1.0 - GAMMA)*0.5;
        kx.s0   *= D_u;    

        kx.s1   += ( v[indices.s0] 
                +    v[indices.s1] ) * GAMMA
                +  ( v[indices.s2] 
                +    v[indices.s3] ) * GAMMA;
        kx.s1   += ( v[indices.s4] 
                +    v[indices.s5] ) * (1.0 - GAMMA)*0.5
                +  ( v[indices.s6] 
                +    v[indices.s7] ) * (1.0 - GAMMA)*0.5;
        kx.s1   *= D_v;
   
        indices = (int8)(  neumannBC((int2)(-2,+0), g, N),
                           neumannBC((int2)(+2,+0), g, N),
                           neumannBC((int2)(+0,-2), g, N),
                           neumannBC((int2)(+0,+2), g, N),
                           get_linear_index(g+(int2)(+1,+0),N),
                           get_linear_index(g+(int2)(-1,+0),N),
                           get_linear_index(g+(int2)(+0,+1),N),
                           get_linear_index(g+(int2)(+0,-1),N) );

        grad_u  = (real2)(u[indices.s0]
                -         u[indices.s1],
                          u[indices.s2]
                -         u[indices.s3]);
        grad_u  +=(real2)(u[indices.s4]
                -         u[indices.s5],
                          u[indices.s6]
                -         u[indices.s7])*8.0;
        grad_u  /= 12.0;    

        grad_v  = (real2)(v[indices.s0]
                -         v[indices.s1],
                          v[indices.s2]
                -         v[indices.s3]);
        grad_v  +=(real2)(v[indices.s4]
                -         v[indices.s5],
                          v[indices.s6]
                -         v[indices.s7])*8.0;
        grad_v  /= 12.0;

        c       = (real2)(c_x - omega*(g.y-origin_y), c_y + omega*(g.x-origin_x));

        kx      += (real2)(dot(c,grad_u), dot(c,grad_v));

        // Add N( x_n ) to existing tangents, yielding kx_n.
        kx      += F(x.s0, x.s1, beta, u_star, k, eps_u, eps_v, D_u, D_v);

        // Bookkeeping.
        ku[gid] =	kx.s0;
        kv[gid] =	kx.s1;
    }
}
__kernel void sol_neumann_dxdt(
     __global real *u,   __global real *v,
     __global real *ku,  __global real *kv,
     real beta, real u_star, real k, 
     real eps_u, real eps_v, real D_u, real D_v,
     real GAMMA, real core_x, real core_y, real c_x, real c_y, real omega, 
     real fwd_adj){
// Computes the rhs of d(u,v)/dt and stores the components in the buffers
// (ku,kv), respectively. 
// First computing D_ij*Laplacian[.], then c[omega].grad(.), 
// and then finally (f).

// The Laplacian stencil is:
//              [(1-g)/2     g     (1-g)/2]
//              [g        -2(1+g)        g]             + O(|h|^4)
//              [(1-g)/2     g     (1-g)/2]
// The gradient stencil is:
//      [1/12     -2/3       0       2/3     -1/12]     + O(|h|^4)
// Combined the stencil is?

     int2 g    = (int2)(get_global_id(0),	get_global_id(1));
     int2 N    = (int2)(get_global_size(0), get_global_size(1));

     int gid   = get_linear_index(g, N);

     real2 x   = (real2)( u[gid], 	 v[gid]);

     real2 kx  = -2.0*(1.0 + GAMMA) * x;

     real2 grad_u, grad_v, c;

// Apply L[.] to (u,v) in turn: 
     kx.s0     += ( u[neumannBC((int2)(+1,+0), g, N)] 
               +    u[neumannBC((int2)(-1,+0), g, N)] ) * GAMMA
               +  ( u[neumannBC((int2)(+0,+1), g, N)] 
               +    u[neumannBC((int2)(+0,-1), g, N)] ) * GAMMA;
     kx.s0     += ( u[neumannBC((int2)(+1,-1), g, N)] 
               +    u[neumannBC((int2)(-1,-1), g, N)] ) * (1.0 - GAMMA)*0.5
               +  ( u[neumannBC((int2)(-1,+1), g, N)] 
               +    u[neumannBC((int2)(+1,+1), g, N)] ) * (1.0 - GAMMA)*0.5;
     kx.s0     *= D_u;    

     kx.s1     += ( v[neumannBC((int2)(+1,+0), g, N)] 
               +    v[neumannBC((int2)(-1,+0), g, N)] ) * GAMMA
               +  ( v[neumannBC((int2)(+0,+1), g, N)] 
               +    v[neumannBC((int2)(+0,-1), g, N)] ) * GAMMA;
     kx.s1     += ( v[neumannBC((int2)(+1,-1), g, N)] 
               +    v[neumannBC((int2)(-1,-1), g, N)] ) * (1.0 - GAMMA)*0.5
               +  ( v[neumannBC((int2)(-1,+1), g, N)] 
               +    v[neumannBC((int2)(+1,+1), g, N)] ) * (1.0 - GAMMA)*0.5;
     kx.s1     *= D_v;

     grad_u    = (real2)(u[neumannBC((int2)(-2,+0), g, N)]
               -         u[neumannBC((int2)(+2,+0), g, N)],
                         u[neumannBC((int2)(+0,-2), g, N)]
               -         u[neumannBC((int2)(+0,+2), g, N)]);
     grad_u    +=(real2)(u[neumannBC((int2)(+1,+0), g, N)]
               -         u[neumannBC((int2)(-1,+0), g, N)],
                         u[neumannBC((int2)(+0,+1), g, N)]
               -         u[neumannBC((int2)(+0,-1), g, N)])*8.0;
     grad_u    /= 12.0;    

     grad_v    = (real2)(v[neumannBC((int2)(-2,+0), g, N)]
               -         v[neumannBC((int2)(+2,+0), g, N)],
                         v[neumannBC((int2)(+0,-2), g, N)]
               -         v[neumannBC((int2)(+0,+2), g, N)]);
     grad_v    +=(real2)(v[neumannBC((int2)(+1,+0), g, N)]
               -         v[neumannBC((int2)(-1,+0), g, N)],
                         v[neumannBC((int2)(+0,+1), g, N)]
               -         v[neumannBC((int2)(+0,-1), g, N)])*8.0;
     grad_v    /= 12.0;

     c         = (real2)(c_x - omega*(g.y-core_y), c_y + omega*(g.x-core_x));

     kx        += (real2)(dot(c,grad_u),dot(c,grad_v));

// Add N( x_n ) to existing tangents, yielding kx_n.
     kx        += F(x.s0, x.s1, beta, u_star, k, eps_u, eps_v, D_u, D_v);

// Bookkeeping.
     ku[gid]   =	kx.s0;
     kv[gid] 	=	kx.s1;
}
__kernel void sol_periodic_dxdt(
     __global real *u,   __global real *v,
     __global real *ku,  __global real *kv,
     real beta, real u_star, real k, 
     real eps_u, real eps_v, real D_u, real D_v,
     real GAMMA, real core_x, real core_y, real c_x, real c_y, real omega, 
     real fwd_adj){
// Computes the rhs of d(u,v)/dt and stores the components in the buffers
// (ku,kv), respectively. 
// First computing D_ij*Laplacian[.], then c[omega].grad(.), 
// and then finally (f).

// The Laplacian stencil is:
//              [(1-g)/2     g     (1-g)/2]
//              [g        -2(1+g)        g]             + O(|h|^4)
//              [(1-g)/2     g     (1-g)/2]
// The gradient stencil is:
//      [1/12     -2/3       0       2/3     -1/12]     + O(|h|^4)
// Combined the stencil is?

     int2 g    = (int2)(get_global_id(0),	get_global_id(1));
     int2 N    = (int2)(get_global_size(0),get_global_size(1));

     int gid   = get_linear_index(g, N);

     real2 x   = (real2)( u[gid], 	 v[gid]);

     real2 kx  = -2.0*(1.0 + GAMMA) * x;

     real2 grad_u, grad_v, c;

// Apply L[.] to (u,v,up,vp) in turn: 
     kx.s0     += (u[periodicBC((int2)(+1,+0), g, N)] 
               +   u[periodicBC((int2)(-1,+0), g, N)] ) * GAMMA
               +  (u[periodicBC((int2)(+0,+1), g, N)] 
               +   u[periodicBC((int2)(+0,-1), g, N)] ) * GAMMA;
     kx.s0     += (u[periodicBC((int2)(+1,-1), g, N)] 
               +   u[periodicBC((int2)(-1,-1), g, N)] ) * (1.0 - GAMMA)*0.5
               +  (u[periodicBC((int2)(-1,+1), g, N)] 
               +   u[periodicBC((int2)(+1,+1), g, N)] ) * (1.0 - GAMMA)*0.5;
     kx.s0     *= D_u;    

     kx.s1     += (v[periodicBC((int2)(+1,+0), g, N)] 
               +   v[periodicBC((int2)(-1,+0), g, N)] ) * GAMMA
               +  (v[periodicBC((int2)(+0,+1), g, N)] 
               +   v[periodicBC((int2)(+0,-1), g, N)] ) * GAMMA;
     kx.s1     += (v[periodicBC((int2)(+1,-1), g, N)] 
               +   v[periodicBC((int2)(-1,-1), g, N)] ) * (1.0 - GAMMA)*0.5
               +  (v[periodicBC((int2)(-1,+1), g, N)] 
               +   v[periodicBC((int2)(+1,+1), g, N)] ) * (1.0 - GAMMA)*0.5;
     kx.s1     *= D_v;

     grad_u    = (real2)(u[periodicBC((int2)(-2,+0), g, N)]
               -         u[periodicBC((int2)(+2,+0), g, N)],
                         u[periodicBC((int2)(+0,-2), g, N)]
               -         u[periodicBC((int2)(+0,+2), g, N)]);
     grad_u    +=(real2)(u[periodicBC((int2)(+1,+0), g, N)]
               -         u[periodicBC((int2)(-1,+0), g, N)],
                         u[periodicBC((int2)(+0,+1), g, N)]
               -         u[periodicBC((int2)(+0,-1), g, N)])*8.0;
     grad_u    /= 12.0;    

     grad_v    = (real2)(v[periodicBC((int2)(-2,+0), g, N)]
               -         v[periodicBC((int2)(+2,+0), g, N)],
                         v[periodicBC((int2)(+0,-2), g, N)]
               -         v[periodicBC((int2)(+0,+2), g, N)]);
     grad_v    +=(real2)(v[periodicBC((int2)(+1,+0), g, N)]
               -         v[periodicBC((int2)(-1,+0), g, N)],
                         v[periodicBC((int2)(+0,+1), g, N)]
               -         v[periodicBC((int2)(+0,-1), g, N)])*8.0;
     grad_v    /= 12.0;

     c         = (real2)(c_x - omega*(g.y-core_y), c_y + omega*(g.x-core_x));

     kx        += (real2)(dot(c,grad_u),dot(c,grad_v));

// Add N( x_n ) to existing tangents, yielding kx_n.
     kx        += F(x.s0, x.s1, beta, u_star, k, eps_u, eps_v, D_u, D_v);

// Bookkeeping.
     ku[gid]   =	kx.s0;
     kv[gid] 	=	kx.s1;
}

// Forward/Adjoint tangent model specific
__kernel void tan_neumann_dxdt(
     __global real *u,   __global real *v,
     __global real *up,  __global real *vp,
     __global real *kup, __global real *kvp, 
     real beta, real u_star, real k, 
     real eps_u, real eps_v, real D_u, real D_v,
     real GAMMA, real core_x, real core_y, real c_x, real c_y, real omega, 
     real fwd_adj){
// Computes the rhs of d(up,vp)/dt and stores the components in the buffers
// (kup,kvp), respectively. 
// First computing D_ij*Laplacian[.], then c[omega].grad(.), 
// and then finally (df).

// The Laplacian stencil is:
//              [(1-g)/2     g     (1-g)/2]
//              [g        -2(1+g)        g]             + O(|h|^4)
//              [(1-g)/2     g     (1-g)/2]
// The gradient stencil is:
//      [1/12     -2/3       0       2/3     -1/12]     + O(|h|^4)
// Combined the stencil is?

     int2 g    = (int2)(get_global_id(0),	get_global_id(1));
     int2 N    = (int2)(get_global_size(0),get_global_size(1));

     int gid   = get_linear_index(g, N);

     real2 xp	= (real2)(up[gid], 	vp[gid]);

     real2 kxp = -2.0*(1.0 + GAMMA) * xp;

     real2 grad_up, grad_vp, c;

// Apply L[.] to (u,v,up,vp) in turn: 

     kxp.s0    += ( up[neumannBC((int2)(+1,+0), g, N)] 
               +    up[neumannBC((int2)(-1,+0), g, N)] ) * GAMMA
               +  ( up[neumannBC((int2)(+0,+1), g, N)] 
               +    up[neumannBC((int2)(+0,-1), g, N)] ) * GAMMA;
     kxp.s0    += ( up[neumannBC((int2)(+1,-1), g, N)] 
               +    up[neumannBC((int2)(-1,-1), g, N)] ) * (1.0 - GAMMA)*0.5
               +  ( up[neumannBC((int2)(-1,+1), g, N)] 
               +    up[neumannBC((int2)(+1,+1), g, N)] ) * (1.0 - GAMMA)*0.5;
     kxp.s0    *= D_u;

     kxp.s1    += ( vp[neumannBC((int2)(+1,+0), g, N)] 
               +    vp[neumannBC((int2)(-1,+0), g, N)] ) * GAMMA
               +  ( vp[neumannBC((int2)(+0,+1), g, N)] 
               +    vp[neumannBC((int2)(+0,-1), g, N)] ) * GAMMA;
     kxp.s1    += ( vp[neumannBC((int2)(+1,-1), g, N)] 
               +    vp[neumannBC((int2)(-1,-1), g, N)] ) * (1.0 - GAMMA)*0.5
               +  ( vp[neumannBC((int2)(-1,+1), g, N)] 
               +    vp[neumannBC((int2)(+1,+1), g, N)] ) * (1.0 - GAMMA)*0.5;
     kxp.s1    *= D_v;

     grad_up   = (real2)(up[neumannBC((int2)(-2,+0), g, N)]
               -         up[neumannBC((int2)(+2,+0), g, N)],
                         up[neumannBC((int2)(+0,-2), g, N)]
               -         up[neumannBC((int2)(+0,+2), g, N)]);
     grad_up   +=(real2)(up[neumannBC((int2)(+1,+0), g, N)]
               -         up[neumannBC((int2)(-1,+0), g, N)],
                         up[neumannBC((int2)(+0,+1), g, N)]
                    -    up[neumannBC((int2)(+0,-1), g, N)])*8.0;
     grad_up   /= 12.0;    

     grad_vp   = (real2)(vp[neumannBC((int2)(-2,+0), g, N)]
               -         vp[neumannBC((int2)(+2,+0), g, N)],
                         vp[neumannBC((int2)(+0,-2), g, N)]
               -         vp[neumannBC((int2)(+0,+2), g, N)]);
     grad_vp   +=(real2)(vp[neumannBC((int2)(+1,+0), g, N)]
               -         vp[neumannBC((int2)(-1,+0), g, N)],
                         vp[neumannBC((int2)(+0,+1), g, N)]
               -         vp[neumannBC((int2)(+0,-1), g, N)])*8.0;
     grad_vp   /= 12.0;

     c         = (real2)(c_x - omega*(g.y-core_y), c_y + omega*(g.x-core_x));

     kxp       += sign(fwd_adj)*(real2)(dot(c,grad_up),dot(c,grad_vp));

     real4 DF  = dF(u[gid], v[gid], beta, u_star, k, eps_u, eps_v, D_u, D_v);

// int fwd_adj \in {-1,1} for adjoint, forward, respectively.
     if( fwd_adj < 0.0 ){ // fwd_adj = sign(dt/dt'), t' a directed parametrization of solution
          DF   = DF.s0213;  // if backwards time, twiddle
     }

// Add N( x_n ) to existing tangents, yielding kx_n.
     kxp       += (real2)(DF.s0 * xp.s0 + DF.s1 * xp.s1, DF.s2 * xp.s0 + DF.s3 * xp.s1);

// Bookkeeping.
     kup[gid]	=	kxp.s0;
     kvp[gid]	=	kxp.s1;
}
__kernel void tan_circle_dxdt(
    __global real *u,   __global real *v,
    __global real *up,  __global real *vp,
    __global real *kup, __global real *kvp, 
    real beta, real u_star, real k, 
    real eps_u, real eps_v, real D_u, real D_v,
    real GAMMA, real origin_x, real origin_y, real c_x, real c_y, real omega, 
    real fwd_adj){
// Computes the rhs of d(u,v,up,vp)/dt and stores the components in the buffers
// (ku,kv,kup,kvp), respectively. 
// First computing D_ij*Laplacian[.], then c[omega].grad(.), 
// and then finally (f,df).

// The Laplacian stencil is:
//              [(1-g)/2     g     (1-g)/2]
//              [g        -2(1+g)        g]             + O(|h|^4)
//              [(1-g)/2     g     (1-g)/2]
// The gradient stencil is:
//      [1/12     -2/3       0       2/3     -1/12]     + O(|h|^4)
// Combined the stencil is?

    int2 g    = (int2)(get_global_id(0),	get_global_id(1));
    int2 N    = (int2)(get_global_size(0),get_global_size(1));
    int gid   = get_linear_index(g, N);
    real r    = length((real2)(g.x,g.y)-(real2)(origin_x,origin_y));
    
    real R    = N.x * 0.5 * 0.75;
    R = fmin(R, fmin(origin_x, N.x-origin_x)); // R < {l1, N1-l1}
    R = fmin(R, fmin(origin_y, N.y-origin_y)); // R < {l2, N2-l2}
     
    
        
        real2 x   = (real2)( u[gid], 	 v[gid]);
        real2 xp  = (real2)(up[gid], 	vp[gid]);

        real2 kxp = -2.0*(1.0 + GAMMA) * xp;

        real2 grad_up, grad_vp, c;
        int8 indices = (int8)(  get_linear_index(g+(int2)(+1,+0),N),
                                get_linear_index(g+(int2)(-1,+0),N),
                                get_linear_index(g+(int2)(+0,+1),N),
                                get_linear_index(g+(int2)(+0,-1),N),
                                get_linear_index(g+(int2)(+1,-1),N),
                                get_linear_index(g+(int2)(-1,-1),N),
                                get_linear_index(g+(int2)(-1,+1),N),
                                get_linear_index(g+(int2)(+1,+1),N) );

        // Apply L[.] to (u,v,up,vp) in turn: 
        kxp.s0  += ( up[indices.s0] 
                +    up[indices.s1] ) * GAMMA
                +  ( up[indices.s2] 
                +    up[indices.s3] ) * GAMMA;
        kxp.s0  += ( up[indices.s4] 
                +    up[indices.s5] ) * (1.0 - GAMMA)*0.5
                +  ( up[indices.s6] 
                +    up[indices.s7] ) * (1.0 - GAMMA)*0.5;
        kxp.s0  *= D_u;

        kxp.s1  += ( vp[indices.s0] 
                +    vp[indices.s1] ) * GAMMA
                +  ( vp[indices.s2] 
                +    vp[indices.s3] ) * GAMMA;
        kxp.s1  += ( vp[indices.s4] 
                +    vp[indices.s5] ) * (1.0 - GAMMA)*0.5
                +  ( vp[indices.s6] 
                +    vp[indices.s7] ) * (1.0 - GAMMA)*0.5;
        kxp.s1  *= D_v;
   
        indices = (int8)(  neumannBC((int2)(-2,+0), g, N),
                           neumannBC((int2)(+2,+0), g, N),
                           neumannBC((int2)(+0,-2), g, N),
                           neumannBC((int2)(+0,+2), g, N),
                           get_linear_index(g+(int2)(+1,+0),N),
                           get_linear_index(g+(int2)(-1,+0),N),
                           get_linear_index(g+(int2)(+0,+1),N),
                           get_linear_index(g+(int2)(+0,-1),N) );

        grad_up = (real2)(up[indices.s0]
                -         up[indices.s1],
                          up[indices.s2]
                -         up[indices.s3]);
        grad_up +=(real2)(up[indices.s4]
                -         up[indices.s5],
                          up[indices.s6]
                -         up[indices.s7])*8.0;
        grad_up /= 12.0;    

        grad_vp = (real2)(vp[indices.s0]
                -         vp[indices.s1],
                          vp[indices.s2]
                -         vp[indices.s3]);
        grad_vp +=(real2)(vp[indices.s4]
                -         vp[indices.s5],
                          vp[indices.s6]
                -         vp[indices.s7])*8.0;
        grad_vp /= 12.0;

        c       = (real2)(c_x - omega*(g.y-origin_y), c_y + omega*(g.x-origin_x));

        kxp     += sign(fwd_adj)*(real2)(dot(c,grad_up),dot(c,grad_vp));

        real4 DF  = dF(x.s0, x.s1, beta, u_star, k, eps_u, eps_v, D_u, D_v);

        // int fwd_adj \in {-1,1} for adjoint, forward, respectively.
        if( fwd_adj < 0.0 ){ // fwd_adj = sign(dt/dt'), t' a directed parametrization of solution
            DF  = DF.s0213;  // if backwards time, twiddle
        }

        // Add N( x_n ) to existing tangents, yielding kx_n.
        kxp     += (real2)(DF.s0 * xp.s0 + DF.s1 * xp.s1, DF.s2 * xp.s0 + DF.s3 * xp.s1);

        // Bookkeeping.
        kup[gid]  =	kxp.s0;
        kvp[gid]	=	kxp.s1;
    if(r > R - 1.0){
      kup[gid]  *= exp(-(r/(R - 1.0)));
      kvp[gid]  *= exp(-(r/(R - 1.0)));
    }
}
__kernel void tan_periodic_dxdt(
     __global real *u,   __global real *v,
     __global real *up,  __global real *vp,
     __global real *kup, __global real *kvp, 
     real beta, real u_star, real k, 
     real eps_u, real eps_v, real D_u, real D_v,
     real GAMMA, real core_x, real core_y, real c_x, real c_y, real omega, 
     real fwd_adj){
// Computes the rhs of d(up,vp)/dt and stores the components in the buffers
// (kup,kvp), respectively. 
// First computing D_ij*Laplacian[.], then c[omega].grad(.), 
// and then finally (df).

// The Laplacian stencil is:
//              [(1-g)/2     g     (1-g)/2]
//              [g        -2(1+g)        g]             + O(|h|^4)
//              [(1-g)/2     g     (1-g)/2]
// The gradient stencil is:
//      [1/12     -2/3       0       2/3     -1/12]     + O(|h|^4)
// Combined the stencil is?

     int2 g    = (int2)(get_global_id(0),	get_global_id(1));
     int2 N    = (int2)(get_global_size(0),get_global_size(1));

     int gid   = get_linear_index(g, N);

     real2 x   = (real2)( u[gid], 	 v[gid]);
     real2 xp  = (real2)(up[gid], 	vp[gid]);

     real2 kxp = -2.0*(1.0 + GAMMA) * xp;

     real2 grad_up, grad_vp, c;

// Apply L[.] to (u,v,up,vp) in turn: 
     kxp.s0    += ( up[periodicBC((int2)(+1,+0), g, N)] 
               +    up[periodicBC((int2)(-1,+0), g, N)] ) * GAMMA
               +  ( up[periodicBC((int2)(+0,+1), g, N)] 
               +    up[periodicBC((int2)(+0,-1), g, N)] ) * GAMMA;
     kxp.s0    += ( up[periodicBC((int2)(+1,-1), g, N)] 
               +    up[periodicBC((int2)(-1,-1), g, N)] ) * (1.0 - GAMMA)*0.5
               +  ( up[periodicBC((int2)(-1,+1), g, N)] 
               +    up[periodicBC((int2)(+1,+1), g, N)] ) * (1.0 - GAMMA)*0.5;
     kxp.s0    *= D_u;

     kxp.s1    += ( vp[periodicBC((int2)(+1,+0), g, N)] 
               +    vp[periodicBC((int2)(-1,+0), g, N)] ) * GAMMA
               +  ( vp[periodicBC((int2)(+0,+1), g, N)] 
               +    vp[periodicBC((int2)(+0,-1), g, N)] ) * GAMMA;
     kxp.s1    += ( vp[periodicBC((int2)(+1,-1), g, N)] 
               +    vp[periodicBC((int2)(-1,-1), g, N)] ) * (1.0 - GAMMA)*0.5
               +  ( vp[periodicBC((int2)(-1,+1), g, N)] 
               +    vp[periodicBC((int2)(+1,+1), g, N)] ) * (1.0 - GAMMA)*0.5;
     kxp.s1    *= D_v;

     grad_up   = (real2)(up[periodicBC((int2)(-2,+0), g, N)]
               -         up[periodicBC((int2)(+2,+0), g, N)],
                         up[periodicBC((int2)(+0,-2), g, N)]
               -         up[periodicBC((int2)(+0,+2), g, N)]);
     grad_up   +=(real2)(up[periodicBC((int2)(+1,+0), g, N)]
               -         up[periodicBC((int2)(-1,+0), g, N)],
                         up[periodicBC((int2)(+0,+1), g, N)]
               -         up[periodicBC((int2)(+0,-1), g, N)])*8.0;
     grad_up   /= 12.0;    

     grad_vp   = (real2)(vp[periodicBC((int2)(-2,+0), g, N)]
               -         vp[periodicBC((int2)(+2,+0), g, N)],
                         vp[periodicBC((int2)(+0,-2), g, N)]
               -         vp[periodicBC((int2)(+0,+2), g, N)]);
     grad_vp   +=(real2)(vp[periodicBC((int2)(+1,+0), g, N)]
               -         vp[periodicBC((int2)(-1,+0), g, N)],
                         vp[periodicBC((int2)(+0,+1), g, N)]
               -         vp[periodicBC((int2)(+0,-1), g, N)])*8.0;
     grad_vp   /= 12.0;

     c         = (real2)(c_x - omega*(g.y-core_y), c_y + omega*(g.x-core_x));

     kxp       += sign(fwd_adj)*(real2)(dot(c,grad_up),dot(c,grad_vp));

     real4 DF  = dF(x.s0, x.s1, beta, u_star, k, eps_u, eps_v, D_u, D_v);

// int fwd_adj \in {-1,1} for adjoint, forward, respectively.
     if( fwd_adj < 0.0 ){ // fwd_adj = sign(dt/dt'), t' a directed parametrization of solution
          DF   = DF.s0213;  // if backwards time, twiddle
     }

// Add N( x_n ) to existing tangents, yielding kx_n.
     kxp       += (real2)(DF.s0 * xp.s0 + DF.s1 * xp.s1, DF.s2 * xp.s0 + DF.s3 * xp.s1);

// Bookkeeping.
     kup[gid]	=	kxp.s0;
     kvp[gid]	=	kxp.s1;
}

// Substep kernels
__kernel void subs(
     __global real *u,   __global real *v,
     __global real *up,  __global real *vp,
     __global real *U,   __global real *V,
     __global real *Up,  __global real *Vp,
     __global real *ku,  __global real *kv,
     __global real *kup, __global real *kvp,
     __global real *du,  __global real *dv,
     __global real *dup, __global real *dvp, 
     real A, real B){
// Takes the current solution (u,v,up,vp) and time-derivative (ku,kv,kup,kvp)
// accumulates the updates (du,dv,dup,dvp) and computes the next subsolution (U,V,Up,Vp).

     size_t gid=   get_global_id(0)                                             // g.x
               +   get_global_size(0) * get_global_id(1);                       // + N.x * g.y

     real4 x   =   (real4)( u[gid],    v[gid],     up[gid],    vp[gid]);
     real4 kx  =   (real4)(ku[gid],   kv[gid],    kup[gid],   kvp[gid]);

// Rescale kx_n by A = (1/6,1/3,1/3,1/6)*dt and add to accumulating update dx.
     du[gid]   += A * kx.s0;
     dv[gid]   += A * kx.s1;
     dup[gid]  += A * kx.s2;
     dvp[gid]  += A * kx.s3;

// Rescale kx_n by B = (1/2,1/2,1/1,0/1)*dt and add to x_n^m to generate x_n^{m+1}.
     x         +=  B * kx;

// Bookkeeping.
     ku[gid]   =   kx.s0;
     kv[gid]   =   kx.s1;
     kup[gid]  =   kx.s2;
     kvp[gid]	 =   kx.s3;
     U[gid]    =   x.s0;
     V[gid]    =   x.s1;
     Up[gid]   =   x.s2;
     Vp[gid]   =   x.s3;
}
__kernel void sol_subs(
     __global real *u,   __global real *v,
     __global real *U,   __global real *V,
     __global real *ku,  __global real *kv,
     __global real *du,  __global real *dv,
     real A, real B){
// Takes the current solution (u,v) and time-derivative (ku,kv)
// accumulates the updates (du,dv) and computes the next subsolution (U,V).

// Convenience.                
     size_t gid=   get_global_id(0)                                             // g.x
               +   get_global_size(0) * get_global_id(1);                       // + N.x * g.y

     real2 x   =   (real2)( u[gid],    v[gid]);
     real2 kx  =   (real2)(ku[gid],   kv[gid]);

// Rescale kx_n by A = (1/6,1/3,1/3,1/6)*dt and add to accumulating update dx.
     du[gid]   += A * kx.s0;
     dv[gid]   += A * kx.s1;

// Rescale kx_n by B = (1/2,1/2,1/1,0/1)*dt and add to x_n^m to generate x_n^{m+1}.
     x         +=  B * kx;

// Bookkeeping.
     ku[gid]   =   kx.s0;
     kv[gid]   =   kx.s1;
     U[gid]    =   x.s0;
     V[gid]    =   x.s1;
}

// Step kernels
__kernel void rk4(	
     __global real *u,   __global real *v,
     __global real *up,  __global real *vp,
     __global real *ku,  __global real *kv,
     __global real *kup, __global real *kvp,
     __global real *du,  __global real *dv,
     __global real *dup, __global real *dvp,
     __global real *U,   __global real *V,
     __global real *Up,  __global real *Vp){

     size_t gid= get_global_id(0)                                               // g.x
               + get_global_size(0) * get_global_id(1);                         // + N.x * g.y

     u[gid]   += du[gid];
     v[gid]	  += dv[gid];
     up[gid]	+= dup[gid];
     vp[gid]	+= dvp[gid];
     U[gid]	  = u[gid];
     V[gid]	  = v[gid];
     Up[gid]	= up[gid];
     Vp[gid]	= vp[gid];
     ku[gid] 	= 0.0;
     kv[gid] 	= 0.0;
     kup[gid] = 0.0;
     kvp[gid]	= 0.0;
     du[gid]  = 0.0;
     dv[gid]  = 0.0;
     dup[gid]	= 0.0;
     dvp[gid]	= 0.0;
}
__kernel void sol_rk4(	
     __global real *u,   __global real *v,
     __global real *ku,  __global real *kv,
     __global real *du,  __global real *dv,
     __global real *U,   __global real *V){

     size_t gid= get_global_id(0)                                               // g.x
               + get_global_size(0) * get_global_id(1);                         // + N.x * g.y

     u[gid]    += du[gid];
     v[gid]    += dv[gid];
     U[gid]	= u[gid];
     V[gid]	= v[gid];
     ku[gid] 	= 0.0;
     kv[gid] 	= 0.0;
     du[gid]   = 0.0;
     dv[gid]   = 0.0;
}
// Interpolation kernels
__kernel void HermiteInterp1( 
     __global real *uu0, __global real *vv0,
     __global real *ku0, __global real *kv0,
     __global real *uu1, __global real *vv1,
     __global real *ku1, __global real *kv1,
     __global real *u0t, __global real *v0t,
     real t, real dt){

     size_t gid= get_global_id(0)                                               // g.x
               + get_global_size(0) * get_global_id(1);                         // + N.x * g.y

     real2 x0  = (real2)(uu0[gid],vv0[gid]);
     real2 x1  = (real2)(uu1[gid],vv1[gid]);
     real2 m0  = (real2)(ku0[gid],kv0[gid]); m0 *= dt;
     real2 m1  = (real2)(ku1[gid],kv1[gid]); m1 *= dt;          
     real2 p   = hermite1(x0,x1,m0,m1,t);

     u0t[gid]  = p.s0;
     v0t[gid]  = p.s1;
}
// Interpolation kernels
__kernel void HermiteInterp2( 
     __global real *uu0, __global real *vv0,
     __global real *ku0, __global real *kv0,
     __global real *uu1, __global real *vv1,
     __global real *ku1, __global real *kv1,
     __global real *u0t, __global real *v0t,
     real t, real dt){

     size_t gid= get_global_id(0)                                               // g.x
               + get_global_size(0) * get_global_id(1);                         // + N.x * g.y

     real2 x0  = (real2)(uu0[gid],vv0[gid]);
     real2 x1  = (real2)(uu1[gid],vv1[gid]);
     real2 m0  = (real2)(ku0[gid],kv0[gid]); m0 *= dt;
     real2 m1  = (real2)(ku1[gid],kv1[gid]); m1 *= dt;          
     real2 p   = hermite2(x0,x1,m0,m1,t);

     u0t[gid]  = p.s0;
     v0t[gid]  = p.s1;
}
// Interpolation kernels
__kernel void HermiteInterp3( 
     __global real *uu0, __global real *vv0,
     __global real *ku0, __global real *kv0,
     __global real *uu1, __global real *vv1,
     __global real *ku1, __global real *kv1,
     __global real *u0t, __global real *v0t,
     real t, real dt){

     size_t gid= get_global_id(0)                                               // g.x
               + get_global_size(0) * get_global_id(1);                         // + N.x * g.y

     real2 x0  = (real2)(uu0[gid],vv0[gid]);
     real2 x1  = (real2)(uu1[gid],vv1[gid]);
     real2 m0  = (real2)(ku0[gid],kv0[gid]); m0 *= dt;
     real2 m1  = (real2)(ku1[gid],kv1[gid]); m1 *= dt;          
     real2 p   = hermite3(x0,x1,m0,m1,t);

     u0t[gid]  = p.s0;
     v0t[gid]  = p.s1;
}
__kernel void HermiteInterp4( 
     __global real *uu0, __global real *vv0, // x_n
     __global real *ku0, __global real *kv0, // f(x_n)
     __global real *uu1, __global real *vv1, // x_{n+1}
     __global real *ku1, __global real *kv1, // f(x_{n+1})
     __global real *ku2, __global real *kv2, // f(u0[eta])
     __global real *u1t, __global real *v1t, // u1[tau = t]
     real t, real dt){                         // t \in (0,1), dt \in R, eta == 1/3

     size_t gid= get_global_id(0)                                               // g.x
               + get_global_size(0) * get_global_id(1);                         // + N.x * g.y

     real2 x0  = (real2)(uu0[gid], vv0[gid]);
     real2 x1  = (real2)(uu1[gid], vv1[gid]);
     real2 m0  = (real2)(ku0[gid], kv0[gid]); m0 *= dt;
     real2 m1  = (real2)(ku1[gid], kv1[gid]); m1 *= dt;
     real2 m2  = (real2)(ku2[gid], kv2[gid]); m2 *= dt; 
     real2 p   = hermite4(x0,x1,m0,m1,m2,t);

     u1t[gid]  = p.s0;
     v1t[gid]  = p.s1;
}
// Amplitude integration
__kernel void area1(__global real *u,   __global real *v,
                    __global real *ku,  __global real *kv,
                    __global real *A,   __global real *B,
                    real dt){
     size_t gid= get_global_id(0)                                               // g.x
               + get_global_size(0) * get_global_id(1);                         // + N.x * g.y
     A[gid]    += dt*ku[gid]*v[gid];
     B[gid]    += dt*kv[gid]*u[gid];             
}
