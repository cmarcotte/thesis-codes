function x = nDTranslateRotate(x, domain, bc, s, thn, nvar, L)

N = prod(domain);
ndim = numel(domain);

%% Generate spatial integration
if ~exist('L', 'var') || isempty(L)
    L = domain;
end
if ~strcmpi(bc, 'periodic')
	NFT = 2*domain;
    L   = 2*L;
    OE = 1;
else
	NFT = domain;
    L   = L;
    OE = 0;
end
addpath ~/Documents/MATLAB/tools/;

diffops = genFourierDiffOps(NFT, L);

shift = mod(s, sign(s).*reshape(L,size(s)));

for nd=1:ndim
    T{nd} = exp(diffops.D{nd} * shift(nd));
end

%% Project U -> R^{N x ndims} x R^{nvars} and displace by s
state = struct([]);
for nv=1:nvar
	
	state{nv} = (reshape(x(1 + (nv-1)*N : nv * N), domain));
	
	state{nv} = fftshift(fft2(mirror(state{nv}, OE)));
	for nd=1:ndim
        if abs(shift(nd)) > eps 
            state{nv} = T{nd} .* state{nv};
        end
	end
    if isreal(x)
        state{nv} = (real(ifftn(ifftshift(state{nv}))));
    else
        state{nv} = ((ifftn(ifftshift(state{nv}))));
    end
	if ~strcmpi(bc, 'periodic')
		state{nv} = state{nv}(1:domain(1), 1:domain(2));
    end
	state{nv} = rot90(state{nv}, round(thn/90));
    
    if abs(thn-round(thn)) > 1e-6 
        state{nv} = imrotate(state{nv},(thn-90*round(thn/90)),'bilinear','crop');
    end
    
	x(1 + (nv-1)*N : nv * N) = reshape((state{nv}), domain);
end

end
