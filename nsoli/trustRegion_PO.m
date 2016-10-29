function [x,mu0,s]=trustRegion(x,v,h,b,f)

	if max(norm(f(x)-x)/norm(b),norm(b)/norm(f(x)-x)) < 1.1
		R = @(x)(norm(f(x)-x));
	else
		R = @(x)(norm(f(x)));
	end
	
	err0 = norm(b);
	[U,d,V] = svd(H,0);
	p = U'*(b'*v);
	d = diag(d);
    z  = p./d; 
    y  = V*z;
    dx = v*y;

tmp = x + dx;
err = R(tmp);
mu0=0;
while (err>err0 && mu0<100)
    if mu0==0
        mu0=1e-4;
    else
        mu0=mu0*1.5;
    end
    z = (p.*d)./(d.^2+mu0);
    y = V*z;
    dx= v*y;
    tmp = x + dx;
    err = R(tmp);
end
x=tmp;
s=1;
end
