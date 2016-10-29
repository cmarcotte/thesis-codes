function Dx0 = dirder_spectral(x0, domain, dim, bc, nvar, L)

N = prod(domain);

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
Dx = diffops.D{1};
Dy = diffops.D{2};

for m=1:nvar
    u{m} = reshape(x0(1 + (m-1)*N : m*N), domain);
    
    U{m} = fftshift(fft2(mirror(u{m}, OE)));

    Dxu{m} = Dx.*U{m};
    Dyu{m} = Dy.*U{m};
    
    Dxu{m} = real(ifft2(ifftshift(Dxu{m})));
    Dyu{m} = real(ifft2(ifftshift(Dyu{m})));
    
    Dxu{m} = Dxu{m}(1:domain(1), 1:domain(2));
    Dyu{m} = Dyu{m}(1:domain(1), 1:domain(2));
end

Dx0 = zeros(size(x0));

if dim == 1
    for m=1:nvar
        Dx0(1 + (m-1)*N : m*N) = Dxu{m}(:);
    end
    
elseif dim == 2
    
    for m=1:nvar
        Dx0(1 + (m-1)*N : m*N) = Dyu{m}(:);
    end
    
end
end
