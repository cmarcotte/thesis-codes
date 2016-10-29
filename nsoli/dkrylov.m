function [step,errstep,total_iters,f_evals, v, h] = dkrylov(x,F,DF,G,params,lmeth,P,vinit)
% Krylov linear equation solver for use in nsoli
%
% function [step, errstep, total_iters, f_evals, v, h]
%                              = dkrylov(x, F, G, params, lmeth, P, vinit)
%
%
% Input:  x = current solution
%         F = nonlinear function
%              the format for f is  function fx = f(x)
%              Note that for Newton-GMRES we incorporate any
%              preconditioning into the function routine.
%         G = Group tangent generators, gx = g(x)
%         params = vector to control iteration
%              params(1) = relative residual reduction factor
%              params(2) = max number of iterations
%              params(3) = max number of restarts for GMRES(m)
%              params(4) (Optional) = reorthogonalization method in GMRES
%                   1 -- Brown/Hindmarsh condition (default)
%                   2 -- Never reorthogonalize (not recommended)
%                   3 -- Always reorthogonalize (not cheap!)
%
%         lmeth = method choice
%              1 GMRES_EQ without restarts (default)
%              2 GMRES_PO without restarts (default)
%              3 Bi-CGSTAB
%              4 TFQMR
%
% Output: x = solution
%         errstep = vector of residual norms for the history of
%                 the iteration
%         total_iters = number of iterations
%

%
% initialization
%

lmaxit = params(2);
restart_limit = 20;
if length(params) >= 3
    restart_limit = params(3);
end
if lmeth == 1, restart_limit = 0; end
if length(params) == 3
    %
    % default reorthogonalization
    %
    gmparms = [params(1), params(2), 1];
elseif length(params) == 4
    %
    % reorthogonalization method is params(4)
    %
    gmparms = [params(1), params(2), params(4)];
else
    gmparms = [params(1), params(2)];
end
%
% linear iterative methods
%
if lmeth == 1
    %
    % compute the step using a GMRES routine
    %
    disp('Using a (relative) equilibria specific solver...');
    [step,errstep,total_iters, v, h] = dgmres_EQ(x,F,DF,G,gmparms,P,vinit);
    kinn = 0;
    %
    % restart at most restart_limit times
    %
    total_iters = total_iters+kinn*lmaxit;
    f_evals = total_iters+kinn;
elseif lmeth == 2
    %
    % compute the step using a GMRES routine
    %
    disp('Using a (relative) periodic orbit specific solver...');
    [step,errstep,total_iters, v, h] = dgmres_PO(x,F,DF,G,gmparms,P,vinit);
    kinn = 0;
    %
    % restart at most restart_limit times
    %
    total_iters = total_iters+kinn*lmaxit;
    f_evals = total_iters+kinn; 
elseif lmeth == 3
    % 
    % Restarted GMRES
    %
    lmaxit = 10;
    disp('Using a (relative) periodic orbit specific solver...');
    step = vinit;
    while total_iters == errstep(total_iters) > gmparms(1)*norm(f0) & ...
                    kinn < restart_limit
        kinn = kinn+1;
        [step, errstep, total_iters, v, h] = dgmres(x, F, DF, G, gmparms, P, step);
    end
    total_iters = total_iters+kinn*lmaxit;
    f_evals = total_iters+kinn;
    %
    % restart at most restart_limit times
    %
    total_iters = total_iters+kinn*lmaxit;
    f_evals = total_iters+kinn;
    %
    % Bi-CGSTAB
    %
elseif lmeth == 4
    [step, errstep, total_iters] = dcgstab(f0, f, x, gmparms);
    f_evals = 2*total_iters;
    %
    % TFQMR
    %
elseif lmeth == 5
    [step, errstep, total_iters] = dtfqmr(f0, f, x, gmparms);
    f_evals = 2*total_iters;
else
    error(' lmeth error in fdkrylov')
end
end
