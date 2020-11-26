function [fk, k, btiters] = ...
    projected_gradient(x0, f, ...
    gradf, kmax, tolgrad, tolx, projectionf, ...
    gamma, alpha0, c1, rho, lsmax, t, type)

xk = x0;
fk = f(xk);
k = 0;

deltaxknorm = tolx + 1;
btiters = [];
flag = 0;

% fin diff
if isempty(gradf)
        disp("Approximation of g")
        h = (10 ^ (-t) .* xk);
        gradf = @(x) finDiff(f, x, h, type);
        flag = 1;
end

pkVal = gradf(xk);

gradnorm = norm(pkVal);

while k < kmax && gradnorm >= tolgrad && deltaxknorm >= tolx
    % selection of the descent direction
    pk = - pkVal;
    % Reset of the value of alpha
    alpha = alpha0;
    
    xbark = xk + gamma * pk;
    
    % application of the projectionf function handle.
    xbark_projected = projectionf(xbark);
    
    % "modified" descent direction
    deltax_projection = (xbark_projected - xk);
    
    xnew = xk + alpha * deltax_projection;
    
    fnew = f(xnew);
    armijo_c1 = c1 * pkVal' * deltax_projection;
    farmijo = fk + armijo_c1 * alpha;
    
    i = 0;
    % Backtracking Iterations (2nd condition is the armijo condition not
    % satisfied) this method is the second for the steplength 
    while i <= lsmax && fnew > farmijo
        % Reduce the value of alpha
        alpha = alpha * rho;
        % Computation of the new x_(k+1) and its value for f
        xnew = xk + alpha * deltax_projection;
        fnew = f(xnew);
        farmijo = fk + armijo_c1 * alpha;
        
        i = i + 1;
    end
    
    % updating
    deltaxknorm = norm(xnew - xk);
    btiters = [btiters; i];
    xk = xnew;
    fk = f(xk);
    gradnorm = norm(pkVal);
    k = k + 1;
    if flag == 1
        h = (10 ^ (-t) * xk);
        gradf = @(x) finDiff(f, x, h, type);
    end
    pkVal = gradf(xk);
end

end
