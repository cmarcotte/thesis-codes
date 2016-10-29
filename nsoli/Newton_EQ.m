function dx = Newton_EQ(x, F, G, v, h, Fx, P, hookstep)   %#ok<*AGROW>
% Computes the correction dx which satisfies 
% DF.dx = -F.x
% For (relative) equilibria, F ~ \partial_{t}x

% Compute sizes, index ranges, and convenient functions
n = numel(x); ng = numel(G);
u_ind = 1:n-ng; g_ind = (n+1-ng):n;

% Compute size of Krylov subspace
k = min(size(h));
hk = h(1:k, 1:k);
vk = v(u_ind, 1:k);

% Compute residual -F(x)
if ~exist('Fx','var') || isempty(Fx)
     Fx = F(x);
end
b = -Fx;

% Compute group tangents at x
for ii=1:ng
    gx(:,ii) = G{ii}(x);
end 

% Compute projection of residual onto group tangents
fprintf('(b,g) = ');
for n=1:ng
    fprintf('%+2.16f\t', (b/norm(b))'*(gx(:,n)/norm(gx(:,n))));
end
fprintf('\n\n');

% Compute Krylov-subspace projections of residual and tangents
bk = vk'*b(u_ind);  
gxk = vk'*gx(u_ind,:);

% Linear solve by least-squares minimization with pseudoinverse
H = [[h; gxk'] [gxk; zeros(ng+1,ng)]];      
y(:,2) = pinv(H) * [v(u_ind,:)'*b(u_ind); zeros(ng,1)];

% Linear solve with a priori marginal mode constraints
H = [[hk; gxk'] [gxk; zeros(ng,ng)]];
y(:,1) = H \ [bk; zeros(ng,1)];

% Linear solve by SVD-hookstep, following channelflow (and Dennis & Schnabel)
if exist('hookstep', 'var')
    [U,D,V] = svd(H);
    delta = min(norm(b), sqrt(eps));
    bh = U'*[bk; zeros(ng,1)];
    mu = 0.; shmu = 0.;
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
            mu = mu - (norm(shmu)/delta) * (Phi/PhiPrime);
        end
     end
     y(:,3) = V*shmu;   
end
% Reconstruct full-space correction from Arnoldi basis
dx(u_ind, 1) = vk*y(1:k,1);         dx(g_ind, 1) = y(k+(1:ng),1);
dx(u_ind, 2) = vk*y(1:k,2);         dx(g_ind, 2) = y(k+(1:ng),2);
if exist('hookstep', 'var')
dx(u_ind, 3) = vk*y(1:k,3);         dx(g_ind, 3) = y(k+(1:ng),3);
end

% Ensure reality
dx = real(dx);

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
for n=1:numel(dx(1,:))
    for m=1:numel(dx(1,:))
        fprintf('%+2.16f ', (dx(:,n)/norm(dx(:,n)))'*(dx(:,m)/norm(dx(:,m))));
    end
    fprintf('\n          ');
end
fprintf('\n');

% Display angle between corrections and group tangents
fprintf('(gx,dx) = ');
for n=1:numel(dx(1,:))
    for m=1:ng
         fprintf('%+2.16f ', (dx(:,n)/norm(dx(:,n)))'*(gx(:,m)/norm(gx(:,m))));
    end
    fprintf('\n          ');
end
fprintf('\n');

% Display angle between corrections and residual
fprintf('(b,dx) = ');
for n=1:numel(dx(1,:))
    fprintf('%+2.16f ', (b/norm(b))'*(dx(:,n)/norm(dx(:,n))));
end
fprintf('\n');

% Decide between them
fprintf('Testing...\n');
for n=1:numel(dx(1,:))
    bb(:,n) = F(x + dx(:,n));
    fprintf('\t|b(%g)|/|b(0)| = %+2.16f,\n', n, norm(bb(:,n))/norm(b));
end
fprintf('\n');
bb = sqrt(sum(bb.^2, 1));
[~,dx_ind] = min(bb);
dx = dx(:, dx_ind);

end
