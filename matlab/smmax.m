% Compute the parameter that results in the largest metric
function [pmax, mmax] = smmax(param, metric, logp, fig)

	if nargin < 3
		logp = false;
	end
	if nargin < 4
		fig = 88;
	end
	
	if logp
		fitparam = log(param);
	else
		fitparam = param;
	end
	fitparam = fitparam(:);
	metric = metric(:);
	
	N = length(fitparam);
	h = diff(fitparam);
	vbar = (metric(2:end) - metric(1:end-1)) ./ h;
	A = [zeros(N-2,1) diag(2*(h(2:end)+h(1:end-1))) zeros(N-2,1)];
	for k = 1:N-2
		A(k,k+2) = h(k);
		A(k,k) = h(k+1);
	end
	b = 3*(h(2:end).*vbar(1:end-1) + h(1:end-1).*vbar(2:end));
	v = fmincon(@(v) JerkCost(v, h, vbar), A \ b, [], [], A, b);
	a = (2./h) .* (3*vbar - 2*v(1:end-1) - v(2:end));
	j = (6./(h.*h)) .* (v(1:end-1) + v(2:end) - 2*vbar);
	
	X = 0;
	Y = 0;
	P = 200;
	for k = 1:N-1
		dx = linspace(0, fitparam(k+1)-fitparam(k), P+1)';
		X(end:end+P) = fitparam(k) + dx;
		Y(end:end+P) = metric(k) + dx.*(v(k) + (dx/2).*(a(k) + dx*j(k)/3));
	end
	
	[mmax, I] = max(Y);
	pmax = X(I);
	if logp
		pmax = exp(pmax);
	end
	
	figure(fig);
	plot(fitparam, metric, 'bx-');
	hold on;
	plot(X, Y, 'r-');
	hold off;
	grid on;

end

function [cost] = JerkCost(v, h, vbar)
	j = (6./(h.*h)) .* (v(1:end-1) + v(2:end) - 2*vbar);
	a = (2./h) .* (3*vbar - 2*v(1:end-1) - v(2:end));
	a(end+1) = a(end) + j(end)*h(end);
	amean = 0.5*sum(h.*(a(1:end-1) + a(2:end))) / sum(h);
	cost = sum((a - amean).^2);
end
% EOF