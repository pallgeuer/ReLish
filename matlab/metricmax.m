% Compute the parameter that results in the largest metric
% [p1, m1] = metricmax([0.5 1 2 4 8], [0.76586 0.78281 0.79250 0.78185 0.69240], false)
% [p2, m2] = metricmax([0.5 1 2 4 8], [0.76586 0.78281 0.79250 0.78185 0.69240], true)
function [pmax, mmax] = metricmax(param, metric, logp, fig)

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
	
	pp = splinefit(fitparam, metric, 2);
	xval = linspace(log(param(1)), log(param(end)), 1001)';
	x = exp(xval);
	if ~logp
		xval = x;
	endif
	mm = ppval(pp, xval);
	[mmax, I] = max(mm);
	pmax = x(I);

	figure(fig);
	plot(param, metric, 'bx-');
	hold on;
	plot(x, mm, 'r-');
	plot(pmax, mmax, 'rx');
	hold off;
	grid on;

end
% EOF