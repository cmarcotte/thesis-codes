function [v , h] = Arnoldi(x, F, DF, fT, v, h, k)
% Computes the next Arnoldi basis vector in the Krylov subspace specified by the current 
% v and h: DF.v(:,k) = h(:,k)'*v(:,k+1), such that v(:,k)'*v(:,j) = KroneckerDelta(i,j).
% The input and output of DF must be real.
v(:,k+1) = DF(x,v(:,k));
%
% Modified Gram-Schmidt
%
for j = 1:k
     h(j,k) = v(:,j)' * v(:,k+1);
     v(:,k+1) = v(:,k+1) - h(j,k) * v(:,j);
end
h(k+1,k) = norm( v(:,k+1) );
%
% Reorthogonalize
%
for j = 1:k
     hr = v(:,j)' * v(:,k+1);
     h(j,k) = h(j,k) + hr;
     v(:,k+1) = v(:,k+1) - hr * v(:,j);
end
h(k+1,k) = norm( v(:,k+1) );

if(abs(h(k+1,k)) > eps)
     v(:,k+1) = v(:,k+1) / h(k+1,k);
     orth_tol = norm( v(:,1:k+1)' * v(:,1:k+1) - eye(k+1), 'fro' );
     if orth_tol > sqrt( numel(v) ) * eps
          fprintf('|v''*v - I| = %2.16f = %g * eps.\n', orth_tol, ceil(orth_tol/eps));
     end
else
     fprintf('Norm of current basis vector is small h(%g,%g) = %+2.16f.\n', k+1, k, h(k+1,k));
     v(:,k+1) = v(:,k+1) / max(h(k+1,k),(k+1)*eps);
end
end
