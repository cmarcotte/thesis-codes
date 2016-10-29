function dx = Newton_PO(x, F, G, v, h, Fx, P, skip_opt)   %#ok<*AGROW>
% Computes the correction dx which satisfies
% (DF - 1).dx = -(F.x - x)
% For (relative) periodic orbits, F ~ x(0) → x(t)

% Compute sizes and index ranges
n = numel(x); ng = numel(G);
u_ind = 1:n-ng; g_ind = (n+1-ng):n;

% Compute size of Krylov subspace
k = min(size(h));
hk = h(1:k, 1:k);
vk = v(u_ind, 1:k);

% Compute residual -(F(x)-x)
if ~exist('Fx','var') || isempty(Fx)
    Fx = F(x);
end
b = x - Fx;
b = P(b);

% Compute group tangents at x and F(x)
for ii=1:ng
    gx(:,ii) = G{ii}(x);
    gFx(:,ii)= G{ii}(Fx);
end

% Compute projection of residual onto group tangents
fprintf('(b,g) = ');
for n=1:ng
    fprintf('%+2.16f\t', (b/norm(b))'*(gx(:,n)/norm(gx(:,n))));
end
fprintf('\n\n');

% Compute Krylov-subspace projections of residual and tangents
bk  = vk'*b(u_ind);
gxk = vk'*gx(u_ind,:);
gFk = vk'*gFx(u_ind,:);

% Prepare the sacrifice (y)
y = zeros(k+ng,1);

% Linear solve with a priori marginal mode constraints (k)
H = [[hk-eye(k); gxk'] [gFk; zeros(ng,ng)]];
y(:,1) = H \ [bk; zeros(ng,1)];

% Linear solve by least-squares minimization with pseudoinverse (k+1)
H = [[h-eye(k+1,k); gxk'] [(v(u_ind,:)'*gFx(u_ind,:)); zeros(ng,ng)]];
y(:,2) = pinv(H) * [v(u_ind,:)'*b(u_ind); zeros(ng,1)];

% Linear solve by least-squares minimization with inverse (k+1)
H = [[h-eye(k+1,k); gxk'] [(v(u_ind,:)'*gFx(u_ind,:)); zeros(ng,ng)]];
y(:,3) = H \ [v(u_ind,:)'*b(u_ind); zeros(ng,1)];

% Trust region model Cauchy-Point y
H = [[h-eye(k+1,k); gxk'] [(v(u_ind,:)'*gFx(u_ind,:)); zeros(ng,ng)]];
dm = -norm(b(u_ind))*H.'*eye(ng+k+1,1);
y(:,4) = dm*(norm(dm)^2/norm(H*dm)^2);

% GMRES point y, for reference
H = [[h-eye(k+1,k); gxk'] [(v(u_ind,:)'*gFx(u_ind,:)); zeros(ng,ng)]];
y(:,5) = (H.'*H)\dm;

% Linear solve by SVD-hookstep, following channelflow (and Dennis & Schnabel)
H = [[hk-eye(k); gxk'] [gFk; zeros(ng,ng)]];
[U,D,V] = svd(H,0);
delta = min(norm(b), sqrt(eps));
bh = U'*[bk; zeros(ng,1)];
mu = 0; shmu = zeros(size(bh));
for n=1:16
    for m=1:numel(bh)
        shmu(m) = shmu(m) + bh(m) / (D(m,m) + mu);
    end
    Phi = shmu'*shmu - delta*delta;
    if (Phi > 1.d-2)
        PhiPrime = 0.;
        for m=1:numel(bh)
            PhiPrime = PhiPrime - 2*bh(m)*bh(m)/((D(m,m) + mu)^3);
        end
        mu = mu + (norm(shmu)/delta) * (Phi/PhiPrime);
    end
end
y(:,6) = V*shmu;

[ee,ll] = eig(hk);
ll = diag(ll);
cs = [];
for n=1:k
    if abs(ll(n)-1.0) < 0.1
        if ~isreal(ll(n))
            if mod(n,2)==0
                cs = [cs; real(ee(:,n).')];
            else 
                cs = [cs; imag(ee(:,n).')];
            end
        else
            cs = [cs; ee(:,n).'];
        end
    end
end
H = [[h-eye(k+1,k); cs] [gFk; zeros(size(cs,1)+1,ng)]];
y(:,7) = pinv(H) * [v(u_ind,:)'*b(u_ind); zeros(size(cs,1),1)];
clear ee ll cs;

% Reconstruct full-space correction from Arnoldi basis
for n=1:size(y,2)
    dx(u_ind, n) = vk*y(1:k,n);
    if n<7
        dx(g_ind, n) = y(k+(1:ng),n);
    end
end

% Ensure reality
dx = real(dx);

% % Remove projection onto group tangents
for n=1:size(dx,2)
    if norm(gx.'*dx(:,n),'inf') > 10*eps
        dx(:,n) = dx(:,n) - gx*((gx.'*gx)\(gx.'*dx(:,n)));
    end
end

% Display group element angle corrections
fprintf('ds = ');
for m=1:ng
    for n=1:numel(dx(1,:))
        fprintf('%+2.16f ', dx(g_ind(m),n));
    end
    fprintf('\n     ');
end
fprintf('\n');

% Display angle between two corrections
fprintf('(dx,dx) = ');
for n=1:size(dx,2)
    for m=1:size(dx,2)
        fprintf('%+2.16f ', (dx(:,n)/norm(dx(:,n)))'*(dx(:,m)/norm(dx(:,m))));
    end
    fprintf('\n          ');
end
fprintf('\n');

% Display angle between corrections and group tangents
fprintf('(gx,dx) = ');
for n=1:size(dx,2)
    for m=1:ng
        fprintf('%+2.16f ', (dx(:,n)/norm(dx(:,n)))'*(gx(:,m)/norm(gx(:,m))));
    end
    fprintf('\n          ');
end
fprintf('\n');

% Display angle between corrections and residual
fprintf('(b,dx) = ');
for n=1:size(dx,2)
    fprintf('%+2.16f ', (b/norm(b))'*(dx(:,n)/norm(dx(:,n))));
end
fprintf('\n');

% Decide between them
if ~exist('skip_opt','var') || isempty(skip_opt) || skip_opt~=1
    fprintf('Testing... \n');
    for n=1:size(dx,2)
        if (n<3) || (n>=3 && min(sqrt(sum(bb(:,1:n-1).^2,1))) >= norm(b))
            bb(:,n) = F(x + dx(:,n)) - (x + dx(:,n));
        else
            bb(:,n) = 2*bb(:,n-1);
        end
        fprintf('\t|b(%g)|/|b(0)| = %+2.16f,\n', n, norm(bb(:,n))/norm(b));
    end
    fprintf('\n');
    bb = abs(sqrt(sum(bb.^2, 1)));
    [~,dx_ind] = min(bb);
else
    dx_ind = 1;
end
dx = dx(:, dx_ind);

end
