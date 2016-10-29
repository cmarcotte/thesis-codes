function A = ocl_amplitude_from_orbit(...
   T, domain, dt, precision, fileName) 

% Continuous (adjoint) tangent
% Keep in mind: dy(1) is given

fid = fopen(fileName,'r');    % open state sequence
fseek(fid,0,-1);              % go back to the beginning
nvar = 2;                     % iunno, in case you forgot
stride = nvar * prod(domain); % stride for reading

N = numel(T);                 % this is how many states there are!

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
area = clkernel('area1',                    global_work_size, local_work_size);

% Allocate
z.u  = clbuffer('rw', precision, int32(prod(domain)));
z.v  = clbuffer('rw', precision, int32(prod(domain)));
z.ku = clbuffer('rw', precision, int32(prod(domain)));
z.kv = clbuffer('rw', precision, int32(prod(domain)));
z.A1 = clbuffer('rw', precision, int32(prod(domain)));
z.A2 = clbuffer('rw', precision, int32(prod(domain)));
% Initialize
null = prec(zeros(1,prod(domain)));
for str=fieldnames(z)'
    z.(char(str)).set(null);
end

% color limits, printing integers, state vectors read in (running tally)
nRead = 0;

%
for n=1:1*N
    % "The adjoint of an explicit Runge-Kutta method is always implicit."
    % So, instead, we're doing Hermite interpolation from x^n -> x^{n+1}
    % and using this function to evaluate L^{+}(x(T-t)) at arbitrary times
    % For this, we need (x^n, dx^n/dt, x^{n+1}, dx^{n+1}/dt).
    % File offsets for later (1) and earlier (2) states
    offset = (mod(n,N)-1)*2*8*stride; % first are [x(N-1),x(N)]
    
    % Read, assign
    fseek(fid, offset, -1);
    x    = fread(fid, stride, 'double'); nRead = nRead + 1;
    kx   = fread(fid, stride, 'double'); nRead = nRead + 1;
    % Assign (x0,k0)
    z.u.set(prec(reshape(x(1:prod(domain)),[1,prod(domain)])));
    z.v.set(prec(reshape(x(1+prod(domain):2*prod(domain)),[1,prod(domain)])));
    z.ku.set(prec(reshape(kx(1:prod(domain)),[1,prod(domain)])));
    z.kv.set(prec(reshape(kx(1+prod(domain):2*prod(domain)),[1,prod(domain)])));
    
    % time-diff
    dT = dt;%T(n)-T(n-1);
   
    area(z.u, z.v, z.ku, z.kv, z.A1, z.A2, dT);
    
    % display
    if ~mod(n, round(N/100))         
        fprintf('%g%% complete.\n',round(100*(n)/N));
    end
end

A = [z.A1.get().'; z.A2.get().'];

% Destroy the evidence
for str=fieldnames(z)';
    z.(char(str)).delete();
end
clear z;
if 2*N ~= nRead
  fprintf('Expected reads = %g. Read = %g.\n', 2*N, nRead);
end
fclose(fid);
end

