function [L,l] = ocl_comoving_karma_jac(...
    x, origin, c, omega, T, p, BC, domain, dt, precision, foradj)

nvar = 2;

if ~exist('foradj','var')
    foradj=1;
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
ocl.addfile('comoving_karma_jac_kernels.cl');
ocl.build();

% Define work sizes
global_work_size    = uint32( domain );
if mod(domain,16) == 0*domain
    local_work_size            = uint32( min(16*ones(size(domain)), domain) ); % 16 * lws = gws
else
    if prod(domain) < 32*32
        local_work_size       = uint32( domain );
    else
        disp('weird domain: no.');
        return;
    end
end

% Define kernel functions
jacobian    = clkernel('jacobian',  global_work_size, local_work_size);

% Allocate
for n=1:nvar
    z.u{n} = clbuffer('rw', precision, int32(prod(domain)));
    z.c{n} = clbuffer('rw', precision, int32(prod(domain)));
    for m=1:nvar
        z.J{n+(m-1)*nvar} = clbuffer('rw', precision, int32(prod(domain)));
    end
end

% Allocate host buffers
u{1}  = real(x(1:prod(domain)));
u{2}  = real(x(1+prod(domain):2*prod(domain)));

% Initialize
null    = prec(zeros(1,prod(domain)));
for str=fieldnames(z)'
    for n=1:numel(z.(char(str)))
        z.(char(str)){n}.set(null);
    end
end
% for str=fieldnames(z)'
%     z.(char(str)).set(null);
% end
z.u{1}.set(	reshape(u{1}, 	[1,prod(domain)]));
z.u{2}.set(	reshape(u{2}, 	[1,prod(domain)]));

% Cast parameters to proper types
for str=fieldnames(p)'
    p.(char(str)) = prec(p.(char(str)));
end

% Generate slices of Jacobian operator
jacobian(z.u{1},z.u{2},...
    z.J{1},z.J{2},z.J{3},z.J{4},...
    z.c{1},z.c{2},...
    p.beta, p.u_star, p.k,...
    p.eps_u, p.eps_v, p.D_u, p.D_v,...
    p.gamma, prec(origin(1)), prec(origin(2)), prec(c(1)), prec(c(2)),...
    prec(omega));

% Recover slices
for n=1:nvar
    C{n} = z.c{n}.get();
    for m=1:nvar
        J{n+(m-1)*nvar} = z.J{n+(m-1)*nvar}.get();
    end
end

% Destroy the buffers
for str=fieldnames(z)';
    for n=1:numel(z.(char(str)))
        z.(char(str)){n}.delete();
    end
end

% free up some memory
clear z u;

% build sparse jacobian matrix
%   D*Laplacian + c.grad + f'
N2 = prod(domain);

%   d/dy
l{1} = spdiags(ones(nvar*N2,1)*[1/12,-2/3,0,2/3,-1/12], [-2,-1,0,1,2], nvar*N2, nvar*N2);
for kk=1:nvar
    for jj=1:domain(2)
        for ii=1:domain(1)
            if (ii==1) || (ii==2) || (ii==(domain(1)-1)) || (ii==domain(1))
                ind = ii + (jj-1)*domain(1) + (kk-1)*N2;
                l{1}(ind,:) = 0;
            end
            if ii==2
                sr = ind + (-1:+2);
                l{1}(ind,sr) = [-7,0,+8,-1]/12;
            end
            if ii==(domain(1)-1)
                sr = ind + (-2:+1);
                l{1}(ind,sr) = [+1,-8,0,+7]/12;
            end
        end
    end
end
%   d/dy -> c_y * d/dy
l{1} = spdiags([C{2}(:);C{2}(:)], 0, nvar*N2, nvar*N2) * l{1};

%   d/dx
l{2} = spdiags(ones(nvar*N2,1)*[-1/12,2/3,0,-2/3,1/12], domain(1)*[-2,-1,0,1,2], nvar*N2, nvar*N2);
for kk=1:nvar
    for jj=1:domain(2)
        for ii=1:domain(1)
            if (jj==1) || (jj==2) || (jj==(domain(2)-1)) || (jj==domain(2))
                ind = ii + (jj-1)*domain(1) + (kk-1)*N2;
                l{2}(ind,:) = 0;
            end
            if jj==2
                sr = ind + (-1:+2)*domain(1);
                l{2}(ind,sr) = [+7,0,-8,+1]/12;
            end
            if jj==(domain(2)-1)
                sr = ind + domain(1)*(-2:+1);
                l{2}(ind,sr) = [-1,+8,0,-7]/12;
            end
        end
    end
end
%   d/dx -> c_x * d/dx
l{2} = spdiags([C{1}(:);C{1}(:)], 0, nvar*N2, nvar*N2) * l{2};

if foradj<0
    l{1} = -l{1};
    l{2} = -l{2};
end

%   d^2/dx^2 + d^2/dy^2
G = p.gamma;
L2 = [  [(1-G)/2,G,(1-G)/2];
        [G,-2*(1+G),G];
        [(1-G)/2,G,(1-G)/2]   ];
l{3} = spdiags([p.D_u*ones(N2,1)*[L2(1,:),L2(2,:),L2(3,:)];p.D_v*ones(N2,1)*[L2(1,:),L2(2,:),L2(3,:)]], [-domain(1)+(-1:1),-1:1,domain(1)+(-1:1)], nvar*N2, nvar*N2);

for kk=1:nvar
    if kk==1; DD = p.D_u; else DD = p.D_v; end;
     for ii=2:domain(1)-1
        jj=1;
        ind = ii + (jj-1)*domain(1) + (kk-1)*N2;
        sr = [-1,0,1,domain(1)-1,domain(1),domain(1)+1]+ind;
        l{3}(ind,:) = 0;
        l{3}(ind,sr) = DD*[(1+G)/2,-2-G,(1+G)/2,(1-G)/2,G,(1-G)/2];
        jj=domain(2);
        ind = ii + (jj-1)*domain(1) + (kk-1)*N2;
        sr = [-1-domain(1),-domain(1),1-domain(1),-1,0,1]+ind;
        l{3}(ind,:) = 0;
        l{3}(ind,sr) = DD*[(1-G)/2,G,(1-G)/2,(1+G)/2,-2-G,(1+G)/2];
    end
    for jj=2:domain(2)-1
        ii=1;
        ind = ii + (jj-1)*domain(1) + (kk-1)*N2;
        sr = [-domain(1)+(0:1),(0:1),domain(1)+(0:1)]+ind;
        l{3}(ind,:) = 0;
        l{3}(ind,sr) = DD*[(1+G)/2,(1-G)/2,-2-G,G,(1+G)/2,(1-G)/2];
        ii=domain(1);
        ind = ii + (jj-1)*domain(1) + (kk-1)*N2;
        sr = [-domain(1)-1,-domain(1),-1,0,domain(1)-1,domain(1)]+ind;
        l{3}(ind,:) = 0;
        l{3}(ind,sr) = DD*[(1-G)/2,(1+G)/2,G,-2-G,(1-G)/2,(1+G)/2];
    end
    ii=1; jj=1;
    ind = ii + (jj-1)*domain(1) + (kk-1)*N2;
    sr = [0,1,domain(1),domain(1)+1]+ind;
    l{3}(ind,:) = 0;
    l{3}(ind,sr) = DD*[-(3+G)/2,(1+G)/2,(1+G)/2,(1-G)/2];
    ii=1; jj=domain(2);
    ind = ii + (jj-1)*domain(1) + (kk-1)*N2;
    sr = [-domain(1),1-domain(1),0,1]+ind;
    l{3}(ind,:) = 0;
    l{3}(ind,sr) = DD*[(1+G)/2,(1-G)/2,-(3+G)/2,(1+G)/2];
    ii=domain(1); jj=1;
    ind = ii + (jj-1)*domain(1) + (kk-1)*N2;
    sr = [-1,0,domain(1)-1,domain(1)]+ind;
    l{3}(ind,:) = 0;
    l{3}(ind,sr) = DD*[(1+G)/2,-(3+G)/2,(1-G)/2,(1+G)/2];
    ii=domain(1); jj=domain(2);
    ind = ii + (jj-1)*domain(1) + (kk-1)*N2;
    sr = [-domain(1)-1,-domain(1),-1,0]+ind;
    l{3}(ind,:) = 0;
    l{3}(ind,sr) = DD*[(1-G)/2,(1+G)/2,(1+G)/2,-(3+G)/2];
end

%   df/du
if foradj>0
    l{4} = spdiags([[J{3}(:);null(:)],[J{1}(:);J{4}(:)],[null(:);J{2}(:)]],[-(N2),0,(N2)], nvar*N2, nvar*N2);
else
	l{4} = spdiags([[J{2}(:);null(:)],[J{1}(:);J{4}(:)],[null(:);J{3}(:)]],[-(N2),0,(N2)], nvar*N2, nvar*N2);
end
% Combine terms
L = l{1}; for n=2:4; L = L + l{n}; end;

end
