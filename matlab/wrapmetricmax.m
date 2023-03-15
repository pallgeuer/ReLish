% Compute the parameters that result in the largest metrics
% best = wrapmetricmax([0.5 1 2 4 8], [0.76586 0.78281 0.79250 0.78185 0.69240])
function [best] = wrapmetricmax(param, metric)
	[M, I] = max(metric);
	[p1, m1] = metricmax(param, metric, false, 90);
	[p2, m2] = metricmax(param, metric, true, 91);
	p3 = (p1 + p2) / 2;
	m3 = (m1 + m2) / 2;
	p4 = eval(sprintf("%.1g", p3));
	best = [param(I) M; p1 m1; p2 m2; p3 m3; p4/1.6 nan; p4 nan; p4*1.6 nan];
end
% EOF