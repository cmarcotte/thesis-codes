function [sol, it_hist, ierr, x_hist, v, h] = nsoli(x, F, DF, G, tol, parms, P, vinit)
%
% NSOLI  Newton-Krylov solver, globally convergent 
%        solver for F(x) = 0 subject to constraints G(x) = 0.
%
% HEAVILY BASED ON THE CODE BY C. T. Kelley, April 27, 2001
% I make no promises, other than "it worked for me, mostly"
%
% function [sol, it_hist, ierr, x_hist] = nsoli(x,F,DF,G,tol,parms,P,vinit)
%
% inputs:
%        initial iterate = x
%        function = F
%		 jacobian-vector product = DF
%		 Constraint functions = G{:}
%        tol = [atol, rtol] relative/absolute
%            error tolerances for the nonlinear iteration
%        parms = [maxit, maxitl, etamax, lmeth, restart_limit]
%            maxit = maximum number of nonlinear iterations
%                default = 40
%            maxitl = maximum number of inner iterations before restart
%                in GMRES(m), m = maxitl 
%                default = 40
%                
%                For iterative methods other than GMRES(m) maxitl
%                is the upper bound on linear iterations.
%
%            |etamax| = Maximum error tolerance for residual in inner
%                iteration. The inner iteration terminates
%                when the relative linear residual is
%                smaller than eta*| F(x_c) |. eta is determined
%                by the modified Eisenstat-Walker formula if etamax > 0.
%                If etamax < 0, then eta = |etamax| for the entire
%                iteration.
%                default: etamax = .9
%
%            lmeth = choice of linear iterative method
%                    1 (GMRES:EQ), 2 (GMRES:PO), 
%                    3 (BICGSTAB), 4 (TFQMR)
%                 default = 1 (GMRES, no restarts)
%
%            restart_limit = max number of restarts for GMRES if
%                    lmeth = 2
%                  default = 20
%
%				Windowing function = P
%
%				initial Krylov vector = vinit
% output:
%        sol = solution
%        it_hist(maxit,3) = l2 norms of nonlinear residuals
%            for the iteration, number of function evaluations,
%            and number of steplength reductions
%        ierr = 0 upon successful termination
%        ierr = 1 if after maxit iterations
%             the termination criterion is not satsified
%        ierr = 2 failure in the line search. The iteration
%             is terminated if too many steplength reductions
%             are taken.
%
%    x_hist = matrix of the entire interation history.
%             The columns are the nonlinear iterates. This
%             is useful for making movies, for example, but
%             can consume way too much storage. This is an
%             OPTIONAL argument. Storage is only allocated
%             if x_hist is in the output argument list.
%
%
%
% internal parameters:
%       debug = turns on/off iteration statistics display as
%               the iteration progresses
%
%       alpha = 1.d-4, parameter to measure sufficient decrease
%
%       sigma0 = .1, sigma1 = .5, safeguarding bounds for the linesearch
%
%       maxarm = 20, maximum number of steplength reductions before
%                    failure is reported
%

%
% Set the debug parameter; 1 turns display on, otherwise off.
%
debug = 1;
format long;
%
% Set internal parameters.
%
alpha = 1.d-4; sigma0 = .1; sigma1 = .5; maxarm = 64; gamma = .9;
%
% Initialize it_hist, ierr, x_hist, and set the default values of
% those iteration parameters which are optional inputs.
%
ierr = 0; maxit = 40; lmaxit = 40; etamax = .9;
it_histx = zeros(maxit,3); lmeth = 1; restart_limit = 20;
if nargout >= 4, x_hist = x; end
%
% Initialize parameters for the iterative methods.
% Check for optional inputs.
%
gmparms = [abs(etamax), lmaxit];

maxit = parms(1); lmaxit = parms(2); etamax = parms(3);
it_histx = zeros(maxit,3);
gmparms = [abs(etamax), lmaxit];

lmeth = parms(4);
%
rtol = tol(2); atol = tol(1); n = numel(x); fnrm = 1; itc = 0; ng = numel(G);
u_ind = 1:n-ng; g_ind = (n+1-ng):n;
%
% Evaluate F at the initial iterate,and
% compute the stop tolerance.
%
if exist('P', 'var') == 0
	P = @(x)(x);
end
fT = F(x);
f0 = fT - (lmeth==2)*x; % lmeth==1: EQ, f0=fT; lmeth==2: PR, f0=fT-x;
fnrm = norm(P(f0));

it_histx(itc+1,1) = fnrm; it_histx(itc+1,2) = 0; it_histx(itc+1,3) = 0;
fnrmo = 1;
stop_tol = atol + rtol*fnrm;
outstat(itc+1, :) = [itc fnrm 0 0 0];
%
% main iteration loop
%
while(fnrm > stop_tol & itc < maxit)

	fprintf('|b|_2 = %2.16f, |b|_2/|u|_inf = %2.16f.\n', ...
	norm(P(f0)), norm(P(f0))/norm((x(u_ind)),'inf'));

%
% Keep track of the ratio (rat = fnrm/frnmo)
% of successive residual norms and 
% the iteration counter (itc).
%
    rat = fnrm/fnrmo;
    fnrmo = fnrm; 
    itc = itc+1;
    
	if exist('vinit', 'var') && all(size(vinit) == size(x)) && itc <= 1
        vinit = vinit;
    else
        vinit = [];
	end

    [dx,errdx,inner_it_count,inner_f_evals, v, h] =...
        dkrylov(x,F,DF,G,gmparms,lmeth,P,vinit);
%
% The line search starts here.
%
% for line_search_iter = 1:10
%   if n>1
%     switch lmeth
%       case 1
%         dx = Newton_EQ(xF,DF,G,v,h,f0,P,1);
%       case 2
%         dx = Newton_PO(xF,DF,G,v,h,f0,P,1);
%      end
%   end
    disp('start line search')
    xold = x;
    lambda = 1; lamm = 1; lamc = lambda; iarm = 0;
    
    xt = x + lambda*dx;
    while isnan(norm(F(xt)-(lmeth==2)*xt))
        lambda = 0.5 * lambda;
        xt = x + lambda * dx;
    end

    ft = F(xt);    
    ft = ft - (lmeth==2)*xt;
    nft = norm(P(ft));
    if isnan(nft) == 1 || isinf(nft) == 1
        nft = norm(xt(1:end-ng).^2);
        disp('|b| was NaN or Inf, FYI. Reset.');
    end
    nf0 = norm(P(f0)); ff0 = nf0*nf0; 
    ffc = nft*nft; ffm = nft*nft;
    while nft >= (1 - alpha*lambda) * nf0;
%
% Apply the three point parabolic model.
%
if iarm == 0
    lambda = sigma1*lambda;
else
    lambda = parab3p(lamc, lamm, ff0, ffc, ffm);
end
if isinf(lambda) == 1 || isnan(lambda) == 1
    lambda = 1.d-6;
end
%
% Update x; keep the books on lambda.
%
        xt = x + lambda*dx;
        lamm = lamc;
        lamc = lambda;
%
% Keep the books on the function norms.
%
        ft = F(xt);
	    ft = P(ft - (lmeth==2)*xt);
        
        nft = norm(ft);
        ffm = ffc;
        ffc = nft*nft;
        iarm = iarm+1;
        if iarm > maxarm || lambda < min(stop_tol / norm(dx), 1.d-10);
            disp('Armijo failure, too many reductions or lambda too small');
            ierr = 2;
            lambda
            iarm
            maxarm
            it_hist = it_histx(1:itc,:);
            if nargout >= 4
                x_hist = [x_hist,x];
            end
            sol = xold;
            return;
        end
    end
    x = xt;
    f0 = ft;
%  end
%
% End of line search.
%    
    if nargout >= 4, x_hist = [x_hist,x]; end
    
    fnrm = norm(f0)
    lambda = lambda
    disp('end of line search')
    
    it_histx(itc+1,1) = fnrm; 

%
% How many function evaluations did this iteration require?
%
    it_histx(itc+1,2) = it_histx(itc,2)+inner_f_evals+iarm+1;

    if itc == 1, it_histx(itc+1,2) = it_histx(itc+1,2)+1; end;

    it_histx(itc+1,3) = iarm;
    
    rat = fnrm/fnrmo;

%
% Adjust eta
%
    if etamax > 0

        etaold = gmparms(1);

        etanew = gamma*rat*rat;

		if gamma*etaold*etaold > .1

            etanew = max(etanew,gamma*etaold*etaold);

		end
        gmparms(1) = min([etanew,etamax]);

        gmparms(1) = max(gmparms(1),.5*stop_tol/fnrm);

    end
    
    etack=gmparms(1);
    outstat(itc+1, :) = [itc fnrm inner_it_count rat iarm];
    
end

sol = x;
if ~exist('v','var') || ~exist('h','var')
    v = [];
    h = [];
end

it_hist = it_histx(1:itc+1,:);
if debug == 1
    disp(outstat)
    it_hist = it_histx(1:itc+1,:);
end
%
% on failure, set the error flag
%
if fnrm > stop_tol
	ierr = 1 
end
end
