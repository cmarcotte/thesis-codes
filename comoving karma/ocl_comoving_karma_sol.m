function [x,T,fileName] = ocl_comoving_karma_sol(...
    x, origin, c, omega, T, p, BC, domain, dt, precision, fileID,...
    integrationOrder,forFile)

if ~exist('integrationOrder','var') || isempty(integrationOrder)
    integrationOrder = 4;
else
%     fprintf('Time integration order: %d.\n',integrationOrder);
end
if exist('forFile','var') && ~isempty(forFile) && norm(imag(x))>0
    tanfid = fopen(forFile,'w');
    fprintf('Saving forward tangent evolution to %s.\n',forFile);
    tansave=1;
    tanwrit=0;
else
    tansave=0;
    tanwrit=0;
end

% time-integration
if numel(T)==1 && T<0
    disp('Incompatible time. Fixing.')
    T = abs(T);
end
if numel(T) == 1
    T = (T/ceil(T/dt))*(0:ceil(T/dt));
end
if T(1) > eps
    fprintf('No, you need T(1)=0.\n');
    return;
end

N = numel(T);
nvar = 2;

stride = nvar * prod(domain);

% Open a file
if exist('fileID','var') && ~isempty(fileID) && nargout>=3
    fileName = strcat('/usr/local/home/cmarcotte3/state_',fileID,'_',...
        num2str(domain(1)),'x',num2str(domain(2)),'x',...
        num2str(nvar),'x',num2str(N),'.dat');
    fid = fopen(fileName,'w');
    saving = 1;
else
    saving = 0;
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
global_work_size               = uint32( domain );
if prod(domain) < 32*32
    local_work_size            = uint32( domain );
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
subs    = clkernel('subs',              global_work_size, local_work_size);
step	= clkernel('rk4',               global_work_size, local_work_size);

% Allocate
z.u	  = clbuffer('rw', precision, int32(prod(domain)));
z.v	  = clbuffer('rw', precision, int32(prod(domain)));
z.up  = clbuffer('rw', precision, int32(prod(domain)));
z.vp  = clbuffer('rw', precision, int32(prod(domain)));
z.ku  = clbuffer('rw', precision, int32(prod(domain)));
z.kv  = clbuffer('rw', precision, int32(prod(domain)));
z.kup = clbuffer('rw', precision, int32(prod(domain)));
z.kvp = clbuffer('rw', precision, int32(prod(domain)));
z.du  = clbuffer('rw', precision, int32(prod(domain)));
z.dv  = clbuffer('rw', precision, int32(prod(domain)));
z.dup = clbuffer('rw', precision, int32(prod(domain)));
z.dvp = clbuffer('rw', precision, int32(prod(domain)));
z.U	  = clbuffer('rw', precision, int32(prod(domain)));
z.V	  = clbuffer('rw', precision, int32(prod(domain)));
z.Up  = clbuffer('rw', precision, int32(prod(domain)));
z.Vp  = clbuffer('rw', precision, int32(prod(domain)));

% Allocate host buffers
u  = real(x(1:prod(domain)));
v  = real(x(1+prod(domain):2*prod(domain)));
up = imag(x(1:prod(domain)));
vp = imag(x(1+prod(domain):2*prod(domain)));
if norm(up(:)) > 1.d-10
    tang = 1;
else
    tang = 0;
end
% Initialize
null    = prec(zeros(1,prod(domain)));
for str=fieldnames(z)'
    z.(char(str)).set(null);
end
z.u.set(	reshape(u, 	[1,prod(domain)]));
z.v.set(	reshape(v, 	[1,prod(domain)]));
z.U.set(	reshape(u,	[1,prod(domain)]));
z.V.set(	reshape(v,	[1,prod(domain)]));
z.up.set(	reshape(up,	[1,prod(domain)]));
z.vp.set(	reshape(vp,	[1,prod(domain)]));
z.Up.set(	reshape(up,	[1,prod(domain)]));
z.Vp.set(	reshape(vp,	[1,prod(domain)]));
% Cast parameters to proper types
for str=fieldnames(p)'
    p.(char(str)) = prec(p.(char(str)));
end
% Integrate
clim = @(g)([0,4]);
try %#ok<*TRYNC>
    addpath /home/cmarcotte3/Documents/MATLAB/sc/;
    set(0, 'CurrentFigure', 1);
    set(1,'color','w');
    state = [z.u.get(),z.v.get()];
    imagesc(reshape(state,[1 2].*domain),clim(state));
    set(gca,'xtick',[],'ytick',[]);
    colormap(jet(1024));
    title(sprintf('$t/T = %g$', 0),'interpreter','latex');
    colorbar();
    truesize();
    drawnow();
end
written = 0;
switch integrationOrder
    % Set of fully explicit low-memory Runge-Kutta methods
    case 1
        % RK1 (Forward Euler):
        A = [1/1]; %#ok<*NBRAK>
        B = circshift([0/1],[0,-1]);
    case 2
        % RK2 (Ralston):
        A = [1/4 3/4];
        B = circshift([0/1 2/3],[0,-1]);
    case 3
        % RK3 (Bogacki-Shampine):
        A = [2/9 3/9 4/9];
        B = circshift([0/1 1/2 3/4],[0,-1]);
    case 4
        % RK4:
        A = [1/6 1/3 1/3 1/6];
        B = circshift([0/1 1/2 1/2 1/1],[0,-1]);
end
for n = 1:N-1
    
    % time-diff
    dT = T(n+1)-T(n);
    a = prec(A*dT);
    b = prec(B*dT);
    state = [z.u.get(),z.v.get()];
    if saving
        count = fwrite(fid,state,precision); written = written + 1;
        if count ~= stride
            disp('Gahhhhhhhhh1!');
            break;
        end
    end
    for nn = 1:integrationOrder
        dxdt(z.U,z.V,z.Up,z.Vp,z.ku,z.kv,z.kup,z.kvp,...
            p.beta, p.u_star, p.k,...
            p.eps_u, p.eps_v, p.D_u, p.D_v,...
            p.gamma, prec(origin(1)), prec(origin(2)), prec(c(1)), prec(c(2)),...
            prec(omega), prec(1.0));
        subs(z.u,z.v,z.up,z.vp,z.U,z.V,z.Up,z.Vp,...
            z.ku,z.kv,z.kup,z.kvp,z.du,z.dv,z.dup,z.dvp,...
            a(nn), b(nn));
        if nn==1 && saving
            kx      = [z.ku.get(),z.kv.get()];
            count   = fwrite(fid,kx,precision);
            written = written + 1;
            if count ~= stride
                disp('Gahhhhhhhhh2!');
                break;
            end
        end
        if nn==1 && tansave
            kp      = [z.Up.get(),z.Vp.get()];
            count   = fwrite(tanfid,kp,precision);
            written = written + 1;
            if count ~= stride
                disp('Gahhhhhhhhh2!');
                break;
            end
        end
    end
    step(z.u, z.v, z.up, z.vp, z.ku, z.kv, z.kup, z.kvp,...
        z.du, z.dv, z.dup, z.dvp, z.U, z.V, z.Up, z.Vp);
    
    if ~mod(n, round(N/100))
        if saving
            fprintf('%g%% complete.\n',round(100*n/N));
        end
        try
            if any(isnan(z.up.get())); disp('nan du'); break; end;
            set(0, 'CurrentFigure', 1);
            if ~tang
                imagesc(reshape(state,[1 2].*domain),clim(state));
                set(gca,'xtick',[],'ytick',[]); colormap(jet(1024));
            elseif tang
                up = [reshape(z.up.get(),domain),reshape(z.vp.get(), domain)];
                imagesc(reshape(up,[1 2].*domain),norm(up(:),'inf')*[-1 +1]);
                set(gca,'xtick',[],'ytick',[]); colormap(redblue(1024));
            end
            colorbar();
            title(sprintf('$t/T = %2.2f$', round(100*n/N)/100),'interpreter','latex');
            truesize();
            drawnow();
        end
    end
end
u = z.u.get(); u = u + 1i*z.up.get(); u = reshape(u,domain);
v = z.v.get(); v = v + 1i*z.vp.get(); v = reshape(v,domain);

% write final state x_N
x(1:nvar*prod(domain)) = [u(:); v(:)];
if saving
    count = fwrite(fid,real(x(1:nvar*prod(domain))),precision); written = written + 1;
    if count ~= 2*prod(domain)
        disp('Gahhhhhhhhh!');
    end
end
if tansave
    count = fwrite(tanfid,imag(x(1:nvar*prod(domain))),precision); written = written + 1;
    if count ~= 2*prod(domain)
        disp('Gahhhhhhhhh!');
    end
end
% evaluate / write final time-derivative f_1^N
dxdt(z.u,z.v,z.up,z.vp,z.ku,z.kv,z.kup,z.kvp,...
    p.beta, p.u_star, p.k,...
    p.eps_u, p.eps_v, p.D_u, p.D_v,...
    p.gamma, prec(origin(1)), prec(origin(2)), prec(c(1)), prec(c(2)),...
    prec(omega), prec(1.0));
if saving
    kx      = [z.ku.get(),z.kv.get()];
    count   = fwrite(fid,kx,precision);
    written = written + 1;
    if count ~= 2*prod(domain)
        disp('Gahhhhhhhhh!');
    end
end
% Column vector
x     = x(:);
% Destroy the evidence
for str=fieldnames(z)';
    z.(char(str)).delete();
end
% verify expected disk writing
if saving
    fprintf('Expected writes = %g. Written = %g.\n', 2*N, written);
    fclose(fid);
end
% free up some memory, close file
clear z;

end
