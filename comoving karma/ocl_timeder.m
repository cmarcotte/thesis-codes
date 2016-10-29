function dx = ocl_timeder(...
    x, origin, c, omega, p, BC, domain, precision, fwd_adj)
% time-derivative
nvar = 2;
if ~exist('fwd_adj','var') || isempty(fwd_adj)
    fwd_adj = 1.0;
end
% Set up OpenCL
addpath /home/cmarcotte3/Documents/opencl-toolbox/
deviceIdx = 1;
ocl = opencl();
ocl.initialize(1,deviceIdx);
switch precision
    case 'single'
        ocl.addfile('prec_32.h');
        prec = @single;
    case 'double'
        ocl.addfile('prec_64.h');
        prec = @double;
end
ocl.addfile('real_numbers.h');
ocl.addfile('comoving_karma_kernels.cl');
ocl.build();

% Define work sizes
global_work_size    = uint32( domain );
if prod(domain) < 32*32
    local_work_size       = uint32( domain );
elseif mod(domain,16) == 0*domain
    local_work_size            = uint32( min(16*ones(size(domain)), domain) ); % 16 * lws = gws
elseif mod(domain,8) == 0*domain
    local_work_size            = uint32( min(8*ones(size(domain)), domain) ); % 16 * lws = gws
elseif mod(domain,4) == 0*domain
    local_work_size            = uint32( min(4*ones(size(domain)), domain) ); % 16 * lws = gws
elseif mod(domain,2) == 0*domain
    local_work_size            = uint32( min(2*ones(size(domain)), domain) ); % 16 * lws = gws
else
    disp('ugh');
    return;
end

% Define kernel functions
dxdt    = clkernel(strcat(BC,'_dxdt'),  global_work_size, local_work_size);

% Allocate
z.u		= clbuffer('rw', precision, int32(prod(domain)));
z.v		= clbuffer('rw', precision, int32(prod(domain)));
z.up    = clbuffer('rw', precision, int32(prod(domain)));
z.vp    = clbuffer('rw', precision, int32(prod(domain)));
z.ku    = clbuffer('rw', precision, int32(prod(domain)));
z.kv    = clbuffer('rw', precision, int32(prod(domain)));
z.kup	= clbuffer('rw', precision, int32(prod(domain)));
z.kvp	= clbuffer('rw', precision, int32(prod(domain)));

% Allocate host buffers
u	    = real(x(1:prod(domain)));
v 	    = real(x(1+prod(domain):2*prod(domain)));
up      = imag(x(1:prod(domain)));
vp      = imag(x(1+prod(domain):2*prod(domain)));

% Initialize
null    = prec(zeros(1,prod(domain)));
z.u.set(	reshape(u, 	[1,prod(domain)]));
z.v.set(	reshape(v, 	[1,prod(domain)]));
z.up.set(	reshape(up,	[1,prod(domain)]));
z.vp.set(	reshape(vp,	[1,prod(domain)]));
z.ku.set(	null);
z.kv.set(	null);
z.kup.set(  null);
z.kvp.set(  null);

% Cast parameters to proper types
for str=fieldnames(p)'
    p.(char(str)) = prec(p.(char(str)));
end  
   
dxdt(z.u,z.v,z.up,z.vp,z.ku,z.kv,z.kup,z.kvp,...
    p.beta, p.u_star, p.k,...
    p.eps_u, p.eps_v, p.D_u, p.D_v,...
    p.gamma, prec(origin(1)), prec(origin(2)), prec(c(1)), prec(c(2)),...
    prec(omega), prec(fwd_adj));


ku = z.ku.get() + 1i*z.kup.get(); 
kv = z.kv.get() + 1i*z.kvp.get();

dx = zeros(nvar*prod(domain),1);

dx(1:nvar*prod(domain),1) = [ku(:);kv(:)];

% Destroy the evidence
for str=fieldnames(z)';
    z.(char(str)).delete();
end

% free up some memory, close file
clear z;

end
