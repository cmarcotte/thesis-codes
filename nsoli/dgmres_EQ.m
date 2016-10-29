function [dx,error,total_iters, v, h] = dgmres_EQ(xc,F,DF,G,params,P,vinit)
% this code bears the scars of use
%
% GMRES linear equation solver for use in Newton-GMRES solver
%
% function [dx, error, total_iters, v, h] = dgmres(xc, F, G, params, P, vinit)
%
%
% Input:  xc = current solution
%         F = nonlinear function e.g., corresponding to the r.h.s. 
%              of \partial_{t}x the format for F is function fx = F(x).
%              Note that for Newton-GMRES we incorporate any
%              preconditioning into the function routine.
%         xc = current point
%         params = two dimensional vector to control iteration
%              params(1) = relative residual reduction factor
%              params(2) = max number of iterations
%              params(3) (Optional) = reorthogonalization method
%                   1 -- Brown/Hindmarsh condition (default)
%                   2 -- Never reorthogonalize (not recommended)
%                   3 -- Always reorthogonalize (not cheap!)
%
%         vinit = initial iterate. vinit = F(xc) is the default. This
%              is a reasonable choice unless the action of G may not 
%              lie in the dynamically accessible tangent space, or 
%              restarted GMRES will be used as the linear solver.
%
% Output: dx = correction to xc: DF.dx = -F(xc)
%         error = vector of residual norms for the history of
%                 the iteration
%         total_iters = number of iterations
%         v = Arnoldi basis for DF
%         h = Arnoldi projection of DF

addpath /home/cmarcotte3/Documents/MATLAB/sc/;

%
% initialization
%
errtol = params(1);
kmax = params(2);

reorth = 1;
if length(params) == 3
    reorth = params(3);
end
f0 = F(xc);
cknrm=norm(f0);

%
% Some convenience for plotting with labels and stuff:
%
IP = {'interpreter','latex'};

%
% The right side of the linear equation for the step is -f0.
%
res = -f0;

x0 = xc;
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
    imagesc(reshape(Pres(u_ind), floor(sqrt(numel(u_ind)/2))*[1 2]));
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
    
    [v,h] = Arnoldi(xc, F, DF, f0, v, h, k);
    
    try
        [e_lyap,lyap] = eig(h(1:k, 1:k));
        lyap = diag(lyap);
        [~,e_ind] = min(abs(lyap-0.0));
        e_lyap = v(u_ind,1:k)*e_lyap(1:k,e_ind(1));
        set(0, 'CurrentFigure', 2);
        subplot(2,2,1);
        e_lyap = reshape(real(e_lyap(u_ind)), floor(sqrt(numel(u_ind)/2))*[1 2]);
        e_lyap = e_lyap/max(max(abs(e_lyap)));
        imagesc(e_lyap,[-1,1]); set(gca, 'XTick', [], 'YTick', []); 
        colormap(redblue(256)); truesize();
        title(sprintf('Re$\\left(\\mathbf{e}_{%g}^{(%g)}\\right)$',e_ind,k),IP{:});
        subplot(2,2,2); cla;
        line([-10 10], [0 0], 'Color', 'k'); hold on;
        line([0 0], [-10 10], 'Color', 'k');
        plot(real(lyap), imag(lyap), 'or', 'MarkerFaceColor', 'w');
        xlim([-3,1]); ylim([-1,1]);
        title('$\lambda$',IP{:});
        hold off;
        subplot(2,2,3);
        vu = reshape(v(u_ind, k+1), floor(sqrt(numel(u_ind)/2))*[1 2]);
        vu = vu / max(max(abs(vu)));
        imagesc(vu, [-1,1]); set(gca, 'XTick', [], 'YTick', []); 
        colormap(redblue(256)); truesize();
        title(sprintf('$\\mathbf{v}_{%g}$', k+1),IP{:});
        subplot(2,2,4); cla;
        plot(1:k, zeros(1, k), '-k');
        hold on;  			% Only care about O(1) stuff
        plot(1:k, (sort(real(lyap), 'descend')), 'or', 'MarkerFaceColor', 'w');
        ylim([-1 1]);
        title('Re$(\lambda)$',IP{:}); hold off;
        drawnow();
    catch
    end
    
    if k > 1 && (mod(k, floor(kmax/4)) == 0)
        
        % Newton step
        
        dx = Newton_EQ(xc, F, G, v, h, [], P);
        x1 = x0 + dx;
        f1 = F(x1);

        errest = norm(P(f0 + DF(xc,dx)));
        err = norm(P(f1));
        fprintf('dx:\n\tlinear approx. = %+2.16f, nonlinear = %+2.16f\n',...
                errest, err);
        
        errck = norm(f1);
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
            if errck < 0.1 * norm(f0)
                x0 = x1;
                disp('Updating state.');
            end
        end
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
dx = Newton_EQ(xc, F, G, v, h, [], P);

f1 = F(x0 + dx);

errest = norm(P(f0 + DF(xc,dx)));
err = norm(P(f1));
fprintf('dx:\n\tlinear approx. = %+2.16f, nonlinear = %+2.16f\n',...
                errest, err);
                
if(isnan(err)==1 || isinf(err)==1)
    disp('final error is inf or NaN')
end
end
