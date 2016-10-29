function [dx,error,total_iters, v, h] = dgmres_PO(xc,F,DF,G,params,P,vinit)
% this code bears the scars of use
%
% GMRES linear equation solver for use in Newton-GMRES solver
%
% function [dx, error, total_iters] = dgmres(f0, F, xc, params, xinit)
%
%
% Input:  f0 = function at current point
%         F = nonlinear function
%              the format for F is  function fx = F(dx)
%              Note that for Newton-GMRES we incorporate any
%              preconditioning into the function routine.
%         xc = current point
%         params = two dimensional vector to control iteration
%              params(1) = relative residual reduction factor
%              params(2) = max number of iterations
%            params(3) (Optional) = reorthogonalization method
%                   1 -- Brown/Hindmarsh condition (default)
%                   2 -- Never reorthogonalize (not recommended)
%                   3 -- Always reorthogonalize (not cheap!)
%
%         xinit = initial iterate. xinit = 0 is the default. This
%              is a reasonable choice unless restarted GMRES
%              will be used as the linear solver.
%
% Output: dx = solution
%         error = vector of residual norms for the history of
%                 the iteration
%         total_iters = number of iterations
%
% Requires dirder.m

addpath /home/cmarcotte3/Documents/MATLAB/tools/;

%
% initialization
%
errtol = params(1);
kmax = params(2);

reorth = 1;
if length(params) == 3
    reorth = params(3);
end
fT = F(xc);
f0 = (fT - xc);
cknrm=norm(f0);

%
% Some cconvenience for plotting with labels and stuff:
IP = {'interpreter','latex'};

%
% The right side of the linear equation for the step is -f0.
%
res = -f0;

xt0 = xc;
%
% Use zero vector as initial iterate for Newton step unless
% the calling routine has a better idea (useful for GMRES(m)).
%
n = numel(xc);
dx = zeros(n,1);
ng = numel(G);
u_ind = 1:n-ng; g_ind = n-ng+1:n;

err = norm(res);

errtol = errtol*err;
error = [];
ksto = [];
%
% Test for termination on entry.
%
total_iters = 0;

if exist('vinit','var') == 1 && isempty(vinit) == 0
    v(:,1) = vinit/norm(vinit);
else
    v(:,1) = res/norm(res);
end
h = [];

k = 0;
Pres = P(res);
try
    set(0, 'CurrentFigure', 2); clf;
    subplot(2,2,3);
    imagesc(reshape(v(u_ind, k+1), floor(sqrt(numel(u_ind)/2))*[1 2])); 
    set(gca, 'xtick', [], 'ytick', []);
    colormap(redblue(256)); truesize();
    title(sprintf('$v_{%g}$', k+1),IP{:});
    subplot(2,2,4);
    imagesc(reshape(Pres(u_ind), floor(sqrt(numel(u_ind)/2))*[1 2]), norm(Pres(u_ind),'inf')*[-1 +1]);
    set(gca, 'xtick', [], 'ytick', []);
    colormap(redblue(256)); colorbar(); truesize();
    title(sprintf('$W_{k}\\delta u$, $|\\delta u| = %2.16f$', norm(Pres(u_ind))),IP{:});
    drawnow();
catch
end
%
% GMRES iteration
%
while ((k < kmax) && (err > errtol))
    k = k+1;
    
    [v,h] = Arnoldi(xc, F, DF, fT, v, h, k);
    
    try
        [e_floquet,floquet] = eig(h(1:k, 1:k));
        floquet = diag(floquet);
        [~,e_ind] = min(abs(floquet-1.0));
        e_floquet = v(u_ind,1:k)*e_floquet(1:k,e_ind(1));
        set(0, 'CurrentFigure', 2);
        subplot(2,2,1);
        %imagesc(h(1:k, 1:k));
        %title(sprintf('$h_{%g, %g} = %g$',k+1,k,h(k+1,k)),IP{:});
        e_floquet = reshape(real(e_floquet(u_ind)), floor(sqrt(numel(u_ind)/2))*[1 2]);
        e_floquet = e_floquet/max(max(abs(e_floquet)));
        imagesc(e_floquet,[-1,1]); set(gca, 'XTick', [], 'YTick', []); 
        colormap(redblue(256)); truesize();
        title(sprintf('Re$\\left(\\mathbf{e}_{%g}^{(%g)}\\right)$',e_ind,k),IP{:});
        subplot(2,2,2);
        theta = 0:pi/32:2*pi;
        plot(real(exp(1i*theta)), imag(exp(1i*theta)), 'k'); hold on;
        line([-1 1], [0 0], 'Color', 'k');
        line([0 0], [-1 1], 'Color', 'k');
        plot(real(floquet), imag(floquet), 'or', 'MarkerFaceColor', 'w');
        axis equal;
        title('$\Lambda$',IP{:});
        hold off;
        subplot(2,2,3);
        vu = reshape(v(u_ind, k+1), floor(sqrt(numel(u_ind)/2))*[1 2]);
        vu = vu / max(max(abs(vu)));
        imagesc(vu, [-1,1]); set(gca, 'XTick', [], 'YTick', []); 
        colormap(redblue(256)); truesize();
        title(sprintf('$\\mathbf{v}_{%g}$', k+1),IP{:});
        subplot(2,2,4);
        semilogy(1:k, ones(1, k), '-k');
        hold on; ylim([0.1 10.0]); 			% Only care about O(1) stuff
        semilogy(1:k, cumprod(sort(abs(floquet), 'descend')), ':k', 'LineWidth', 2);
        semilogy(1:k, (sort(abs(floquet), 'descend')), 'or', 'MarkerFaceColor', 'w');
        title('$|\Lambda|$, $\Pi_{i=1}^{k}|\Lambda_i|$',IP{:}); hold off;
        drawnow();
    catch
    end
    
    if k > 3 && (mod(k-3, floor(kmax/4)) == 0)
        
        %% Newton step
        dx = Newton_PO(xc, F, G, v, h, fT, P);
        
        xt1 = xt0 + dx;
        ft1 = F(xt1);
        errck = norm(ft1-xt1);
        errest = norm(P(f0 + DF(xc,dx) - dx));
        
        fprintf('dx:\n\tlinear approx. = %+2.16f, nonlinear = %+2.16f\n',...
                errest, errck);
        
        if errck < 0.05 * norm(f0)
            break;
        end
        if errck < err
            fprintf('|F(x + dx)| = %+2.16f\n\n', errck);
        end
        if(isnan(errck)==1 || isinf(errck)==1)
            
            err=1e6;
            error=[error;err];
            ksto=[ksto;k];
            
        else
            err=errck;
            error=[error;err];
            ksto=[ksto;k];
            if errck < 0.2 * norm(f0)
                xt0 = xt1;
                disp('Updating state.');
            end
        end
        
        %        else
        %            disp('Tck < 0')
        %            fprintf('Tck = %2.16f\n', Tck);
        %
        %            err=1e6;
        %            error=[error;err];
        %            ksto=[ksto;k];
        %        end
    end
end

%
% At this point either k > kmax or error < errtol.
% It's time to compute dx and leave.
%
if isempty(error) == 0
    [minerr,ik]=min(error);
    k0=ksto(ik);
else
    minerr = norm(res);
    k0 = kmax;
end

%% Newton step
dx = Newton_PO(xc, F, G, v, h, fT, P);

xt1 = xt0 + dx;

ft1 = F(xt1);
err = norm(P(ft1-xt1));
errest = norm(P(f0 + DF(xc,dx) - dx));
        
        fprintf('dx:\n\tlinear approx. = %+2.16f, nonlinear = %+2.16f\n',...
                errest, errck);
                
if(isnan(err)==1 || isinf(err)==1)
    disp('final error is inf or NaN')
    
end

end

