function [U,B,V] = gkl(R,L,U,B,V,k)
	% alpha(i) -> B(i,i)
	%  beta(i) -> B(i,i+1)

		U(:,k) = R(V(:,k));
		if k>1
			U(:,k) = U(:,k) - U(:,k-1)*B(k-1,k);
		end
        B(k,k) = norm(U(:,k));
		U(:,k) = U(:,k)/B(k,k);
		V(:,k+1) = L(U(:,k));
        V(:,k+1) = V(:,k+1) - V(:,k)*B(k,k);
		B(k,k+1) = norm(V(:,k+1));           
		V(:,k+1) = V(:,k+1)/B(k,k+1);
end