function dw = ocl_comoving_karma_adj(...
    dw, origin, c, omega, T, p, BC, domain, dt, precision, fileName,...
    integrationOrder, interpolationOrder, adjFile) %#ok<INUSL>

% Continuous (adjoint) tangent
% Keep in mind: dy(1) is given

fid = fopen(fileName,'r');    % open state sequence
fseek(fid,0,-1);              % go back to the beginning
nvar = 2;                     % iunno, in case you forgot
stride = nvar * prod(domain); % stride for reading

if exist('adjFile','var') && ~isempty(adjFile)
  fid_adj = fopen(adjFile,'w');
  fprintf('Saving adjoint evolution to %s.\n',adjFile);
  saving = 1;
  written = 0;
else
    saving = 0;
    written = 0;
end

if saving
  fwrite(fid_adj,dw,precision); 
  written = written + 1;
end

N = numel(T);                 % this is how many states there are!

if ~exist('integrationOrder','var') || isempty(integrationOrder)
    integrationOrder = 4;
end
if ~exist('interpolationOrder','var') || isempty(interpolationOrder)
    interpolationOrder = integrationOrder;
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
if mod(domain,16) == 0*domain
    local_work_size            = uint32( min(16*ones(size(domain)), domain) ); % 16 * lws = gws
elseif mod(domain,8) == 0*domain
    local_work_size            = uint32( min(8*ones(size(domain)), domain) ); % 16 * lws = gws
else
    if prod(domain) < 32*32
        local_work_size       = uint32( domain );
    else
        disp('weird domain: no.');
        return;
    end
end

% Define kernel functions
hrm1 = clkernel('HermiteInterp1',  global_work_size, local_work_size);
hrm2 = clkernel('HermiteInterp2',  global_work_size, local_work_size);
hrm3 = clkernel('HermiteInterp3',  global_work_size, local_work_size);
hrm4 = clkernel('HermiteInterp4',  global_work_size, local_work_size);

dydt = clkernel(strcat('sol_',BC,'_dxdt'),  global_work_size, local_work_size);
dxdt = clkernel(strcat('tan_',BC,'_dxdt'),  global_work_size, local_work_size);
subs = clkernel('sol_subs',                 global_work_size, local_work_size);
step = clkernel('sol_rk4',                  global_work_size, local_work_size);

% Allocate
z.u0  = clbuffer('rw', precision, int32(prod(domain)));
z.v0  = clbuffer('rw', precision, int32(prod(domain)));
z.u1  = clbuffer('rw', precision, int32(prod(domain)));
z.v1  = clbuffer('rw', precision, int32(prod(domain)));
z.u2  = clbuffer('rw', precision, int32(prod(domain)));
z.v2  = clbuffer('rw', precision, int32(prod(domain)));
z.ku0	= clbuffer('rw', precision, int32(prod(domain)));
z.kv0	= clbuffer('rw', precision, int32(prod(domain)));
z.ku1	= clbuffer('rw', precision, int32(prod(domain)));
z.kv1	= clbuffer('rw', precision, int32(prod(domain)));
z.ku2	= clbuffer('rw', precision, int32(prod(domain)));
z.kv2	= clbuffer('rw', precision, int32(prod(domain)));
z.U   = clbuffer('rw', precision, int32(prod(domain)));
z.V   = clbuffer('rw', precision, int32(prod(domain)));
z.up  = clbuffer('rw', precision, int32(prod(domain)));
z.vp  = clbuffer('rw', precision, int32(prod(domain)));
z.Up  = clbuffer('rw', precision, int32(prod(domain)));
z.Vp  = clbuffer('rw', precision, int32(prod(domain)));
z.kup	= clbuffer('rw', precision, int32(prod(domain)));
z.kvp	= clbuffer('rw', precision, int32(prod(domain)));
z.dup	= clbuffer('rw', precision, int32(prod(domain)));
z.dvp	= clbuffer('rw', precision, int32(prod(domain)));
% Initialize
null = prec(zeros(1,prod(domain)));
for str=fieldnames(z)'
    z.(char(str)).set(null);
end
z.up.set(prec(reshape((dw(1:prod(domain))),[1,prod(domain)])));
z.vp.set(prec(reshape((dw(1+prod(domain):nvar*prod(domain))),[1,prod(domain)])));
z.Up.set(prec(reshape((dw(1:prod(domain))),[1,prod(domain)])));
z.Vp.set(prec(reshape((dw(1+prod(domain):nvar*prod(domain))),[1,prod(domain)])));
% Make parameters correct types
for str=fieldnames(p)'
    p.(char(str)) = prec(p.(char(str)));
end
% color limits, printing integers, state vectors read in (running tally)
clim=@(g)(max(abs(g(:)))*[-1,+1]);
nRead = 0;
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
C = circshift(B,[0,+1]); % interpolant time / dT
%
for n=N:-1:2
    % "The adjoint of an explicit Runge-Kutta method is always implicit."
    % So, instead, we're doing Hermite interpolation from x^n -> x^{n+1}
    % and using this function to evaluate L^{+}(x(T-t)) at arbitrary times
    % For this, we need (x^n, dx^n/dt, x^{n+1}, dx^{n+1}/dt).
    % File offsets for earlier (1) and later (2) states
    offset = [n-2,n-1]*2*8*stride; % first are [x(N-1),x(N)]
    
    % Read, assign
    if n==N;
        % If x is not already assigned, load it.
        % If n < N, then the state is present in x, due to prior step
        fseek(fid, offset(2), -1);
        x    = fread(fid, stride, 'double'); nRead = nRead + 1;
        kx   = fread(fid, stride, 'double'); nRead = nRead + 1;
    end
    % Assign (x0,k0)
    z.u1.set(prec(reshape(x(1:prod(domain)),[1,prod(domain)])));
    z.v1.set(prec(reshape(x(1+prod(domain):2*prod(domain)),[1,prod(domain)])));
    z.ku1.set(prec(reshape(kx(1:prod(domain)),[1,prod(domain)])));
    z.kv1.set(prec(reshape(kx(1+prod(domain):2*prod(domain)),[1,prod(domain)])));
    % Read, assign
    fseek(fid, offset(1), -1);
    x    = fread(fid, stride, 'double'); nRead = nRead + 1;
    kx   = fread(fid, stride, 'double'); nRead = nRead + 1;
    % Assign (x1,k1)
    z.u0.set(prec(reshape(x(1:prod(domain)),[1,prod(domain)])));
    z.v0.set(prec(reshape(x(1+prod(domain):2*prod(domain)),[1,prod(domain)])));
    z.ku0.set(prec(reshape(kx(1:prod(domain)),[1,prod(domain)])));
    z.kv0.set(prec(reshape(kx(1+prod(domain):2*prod(domain)),[1,prod(domain)])));
    % time-diff
    dT = T(n)-T(n-1);
    a = prec(dT * A);
    b = prec(dT * B);
    % Interpolate between x(n-1) -> x(n), create function (t) -> (x(n-1 + t))
    switch interpolationOrder
        case 1
            stateFunc = @(zzu,zzv,tt)(hrm1(z.u0,z.v0,z.ku0,z.kv0,z.u1,z.v1,z.ku1,z.kv1,zzu,zzv,...
                prec(tt),prec(dT)));
        case 2
            if n==N
                disp('No quadratic interpolant for the general case, elevating to cubic.');
            end
            stateFunc = @(zzu,zzv,tt)(hrm2(z.u0,z.v0,z.ku0,z.kv0,z.u1,z.v1,z.ku1,z.kv1,zzu,zzv,...
                prec(tt),prec(dT)));
        case 3
            stateFunc = @(zzu,zzv,tt)(hrm3(z.u0,z.v0,z.ku0,z.kv0,z.u1,z.v1,z.ku1,z.kv1,zzu,zzv,...
                prec(tt),prec(dT)));
        case 4
            % Compute cubic-interpolated state x2 and derivative k2 from
            % (x0,k0,x1,k1) evaluated at eta==1/3
            hrm3(z.u0,z.v0,z.ku0,z.kv0,z.u1,z.v1,z.ku1,z.kv1,z.u2,z.v2,...
                prec(1.0/3.0),prec(dT));
            dydt(z.u2,z.v2,z.ku2,z.kv2,...
                p.beta, p.u_star, p.k,...
                p.eps_u, p.eps_v, p.D_u, p.D_v,...
                p.gamma, prec(origin(1)), prec(origin(2)),prec(c(1)),prec(c(2)),...
                prec(omega), prec(1.0));
            stateFunc = @(zzu,zzv,tt)(hrm4(z.u0, z.v0, z.ku0, z.kv0, ...
                z.u1, z.v1, z.ku1, z.kv1, ...
                z.ku2,z.kv2, zzu, zzv, ...
                prec(tt), prec(dT)));
    end
    
    for nn=1:integrationOrder
        % Compute interpolated state at time T^{n} - t = T^{n} + t - dT
        t = prec(1.0 - C(nn));
        stateFunc(z.U,z.V,t);
%         dxdt(z.U,z.V,z.up,z.vp,z.kup,z.kvp,...
%             p.beta, p.u_star, p.k,...
%             p.eps_u, p.eps_v, p.D_u, p.D_v,...
%             p.gamma, prec(origin(1)), prec(origin(2)),prec(c(1)),prec(c(2)),...
%             prec(omega), prec(-1.0));
        dxdt(z.U,z.V,z.Up,z.Vp,z.kup,z.kvp,...
            p.beta, p.u_star, p.k,...
            p.eps_u, p.eps_v, p.D_u, p.D_v,...
            p.gamma, prec(origin(1)), prec(origin(2)),prec(c(1)),prec(c(2)),...
            prec(omega), prec(-1.0));
        subs(z.up,z.vp,z.Up,z.Vp,z.kup,z.kvp,z.dup,z.dvp,...
            a(nn),b(nn));
    end
    
    % update xp, assign Xp, clear kxp, clear dxp
    step(z.up, z.vp, z.kup, z.kvp, z.dup, z.dvp, z.Up, z.Vp);
    
    if saving
      dw = [z.up.get().'; z.vp.get().'];
      fwrite(fid_adj,dw,precision); 
      written = written + 1;
    end
    
    % display
    if ~mod(n, round(N/100))
        try
            if any(isnan(z.up.get())); disp('nan du'); break; end;
            set(0, 'CurrentFigure', 1);
            xp = [reshape(z.up.get(),domain),reshape(z.vp.get(),domain)];
            imagesc(xp,clim(xp)); set(gca,'xtick',[],'ytick',[]);
            colormap(puwhgr(256)); colorbar();
            title(sprintf('$t/T = %g$', round(100*n/N)/100),'interpreter','latex');
            truesize();
            drawnow();
        catch
          fprintf('%g%% complete.\n',round(100*(N-n)/N));
        end
    end
end

dw = [z.up.get().'; z.vp.get().'];

% Destroy the evidence
for str=fieldnames(z)';
    z.(char(str)).delete();
end
clear z;
if 2*N ~= nRead
  fprintf('Expected reads = %g. Read = %g.\n', 2*N, nRead);
end
fclose(fid);
if saving
  if N ~= written
    fprintf('Expected writes = %g. Written = %g.\n', N, written);
  end
  fclose(fid_adj);
end

end