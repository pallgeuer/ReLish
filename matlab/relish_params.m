% Determine (alpha, beta) pairs for ReLish activation function options
%
% C2 = Whether the activation function should be C2 (true) or just C1 (false)
% alpha_range = Logarithmic [min max] range to draw alpha values from
% alpha_num = Number of alpha values to consider
% beta_range = Logarithmic [min max] range to draw beta values from
% beta_num = Number of beta values to consider
% gamma_range = Linear [min max] range to use for gamma
% gamma_num = Number of gamma values to use
% similarity_thres = Minimum Gaussian-weighted (stddev=3) difference between selected activation functions
% area_range = Allowable area range
% xmin_range = Allowable xmin range
% fmin_range = Allowable fmin range
% plot_all = Whether to animate a plot with all candidates
%
% Example:
%   chosen = relish_params(false, [0.01 10.0], 101, [0.1 10.0], 101, [1 1.75], 4, 0.035, [0 2], [-2 0], [-1 0], false);
%   T = array2table(chosen); T.Properties.VariableNames(1:3) = {'alpha', 'beta', 'gamma'}; writetable(T, 'relishc1.csv');
%   fid = fopen('relishc1.txt','w'); fprintf(fid, '[''relu'''); for params = chosen(randperm(size(chosen,1)),:)'; fprintf(fid, ', ''relishg1-%.4g-%.4g-%.4g''', params(1), params(2), params(3)); end; fprintf(fid, ']\n');
%   chosen = relish_params(true, [0.01 10.0], 101, [0.1 10.0], 101, [1 1.75], 4, 0.035, [0 2], [-2 0], [-1 0], false);
%   T = array2table(chosen); T.Properties.VariableNames(1:3) = {'alpha', 'beta', 'gamma'}; writetable(T, 'relishc2.csv');
%   fid = fopen('relishc2.txt','w'); fprintf(fid, '[''relu'''); for params = chosen(randperm(size(chosen,1)),:)'; fprintf(fid, ', ''relishg2-%.4g-%.4g-%.4g''', params(1), params(2), params(3)); end; fprintf(fid, ']\n');
%
function [chosen, candidates] = relish_params(C2, alpha_range, alpha_num, beta_range, beta_num, gamma_range, gamma_num, similarity_thres, area_range, xmin_range, fmin_range, plot_all)

	% Default arguments
	if nargin < 9 || isempty(area_range)
		area_range = [0 inf];
	end
	if nargin < 10 || isempty(xmin_range)
		xmin_range = [-inf 0];
	end
	if nargin < 11 || isempty(fmin_range)
		fmin_range = [-inf 0];
	end
	if nargin < 12 || isempty(plot_all)
		plot_all = false;
	end

	% Select which type of ReLish activation function
	if C2
		f = @(x, alpha, beta) x .* alpha ./ (2*cosh(beta*x) + alpha - 2);
	else
		f = @(x, alpha, beta) x .* alpha ./ (exp(-beta*x) + alpha - 1);
	end

	% Mish and swish activation functions
	mish = @(x) x .* tanh(log(1 + exp(x)));
	swish = @(x) x ./ (1 + exp(-x));
	mishgc = @(x) mish(x) / 0.6;  % Scaled to have gradient 1 at x = 0
	swishgc = @(x) swish(x) / 0.5;  % Scaled to have gradient 1 at x = 0

	% Optimisation options
	O = optimset('Display', 'notify-detailed', 'MaxFunEvals', 1000);

	% Calculate minimum points of mish and swish
	[xminmish, fminmish] = MinimumPoint(O, mish);
	[xminswish, fminswish] = MinimumPoint(O, swish);

	% Calculate parameters that have the minimum point closest to the mish/swish minimum points
	pmishpm = MinimumPointMatchingCost(f, xminmish, fminmish, O);
	pswishpm = MinimumPointMatchingCost(f, xminswish, fminswish, O);

	% Set up x space
	xparam = linspace(-20, 0, 10001)';
	dx = (xparam(end) - xparam(1)) / (length(xparam) - 1);

	% Calculate parameters that minimise unit-gaussian weighted norm-difference to mish/swish
	weight = exp(-0.5*xparam.^2);  % Standard deviation of 1
	weighttotal = dx * (sum(weight) - (weight(1) + weight(end))/2);
	fparammish = mish(xparam);
	fparamswish = swish(xparam);
	fparammishgc = mishgc(xparam);
	fparamswishgc = swishgc(xparam);
	pmishcm = MinimumCurveMatching(f, xparam, fparammish, dx, weight, weighttotal, O);
	pswishcm = MinimumCurveMatching(f, xparam, fparamswish, dx, weight, weighttotal, O);
	pmishgccm = MinimumCurveMatching(f, xparam, fparammishgc, dx, weight, weighttotal, O);
	pswishgccm = MinimumCurveMatching(f, xparam, fparamswishgc, dx, weight, weighttotal, O);

	% Calculate parameters that minimise unit-gaussian weighted norm-difference derivative to mish/swish
	dweight = exp(-0.5*((xparam(1:end-1) + xparam(2:end))/4).^2);  % Standard deviation of 2
	dweighttotal = dx * sum(dweight);
	dfparammish = diff(fparammish);
	dfparamswish = diff(fparamswish);
	dfparammishgc = diff(fparammishgc);
	dfparamswishgc = diff(fparamswishgc);
	pmishdm = MinimumDerivMatching(f, xparam, dfparammish, dx, dweight, dweighttotal, O);
	pswishdm = MinimumDerivMatching(f, xparam, dfparamswish, dx, dweight, dweighttotal, O);
	pmishgcdm = MinimumDerivMatching(f, xparam, dfparammishgc, dx, dweight, dweighttotal, O);
	pswishgcdm = MinimumDerivMatching(f, xparam, dfparamswishgc, dx, dweight, dweighttotal, O);

	% Calculate the function characterisations for a grid of alpha/beta values
	alpha_value = logspace(log10(alpha_range(1)), log10(alpha_range(2)), alpha_num)';
	beta_value = logspace(log10(beta_range(1)), log10(beta_range(2)), beta_num)';
	[A, B] = meshgrid(alpha_value, beta_value);

	% Add default heuristic parameter sets
	A = [1; 2; pmishpm(1); pswishpm(1); pmishcm(1); pswishcm(1); pmishgccm(1); pswishgccm(1); pmishdm(1); pswishdm(1); pmishgcdm(1); pswishgcdm(1); A(:)];
	B = [1; 1; pmishpm(2); pswishpm(2); pmishcm(2); pswishcm(2); pmishgccm(2); pswishgccm(2); pmishdm(2); pswishdm(2); pmishgcdm(2); pswishgcdm(2); B(:)];
	N = length(A);

	% Calculate the candidate parameters
	candidates = [A B zeros(N, 11)];  % alpha beta area xmin fmin mishdiff swishdiff mishgcdiff swishgcdiff dmishdiff dswishdiff dmishgcdiff dswishgcdiff
	for n = 1:N
		alpha = candidates(n,1);
		beta = candidates(n,2);
		fparam = f(xparam, alpha, beta);
		dfparam = diff(fparam);
		candidates(n,3) = -dx * (sum(fparam) - (fparam(1) + fparam(end))/2);
		[xmin, fmin] = MinimumPoint(O, f, alpha, beta);
		candidates(n,4) = xmin;
		candidates(n,5) = fmin;
		candidates(n,6) = sqrt(CurveMatchingCostFP(fparam, fparammish, dx, weight, weighttotal));
		candidates(n,7) = sqrt(CurveMatchingCostFP(fparam, fparamswish, dx, weight, weighttotal));
		candidates(n,8) = sqrt(CurveMatchingCostFP(fparam, fparammishgc, dx, weight, weighttotal));
		candidates(n,9) = sqrt(CurveMatchingCostFP(fparam, fparamswishgc, dx, weight, weighttotal));
		candidates(n,10) = sqrt(DerivMatchingCostFP(dfparam, dfparammish, dx, dweight, dweighttotal));
		candidates(n,11) = sqrt(DerivMatchingCostFP(dfparam, dfparamswish, dx, dweight, dweighttotal));
		candidates(n,12) = sqrt(DerivMatchingCostFP(dfparam, dfparammishgc, dx, dweight, dweighttotal));
		candidates(n,13) = sqrt(DerivMatchingCostFP(dfparam, dfparamswishgc, dx, dweight, dweighttotal));
	end

	% Set up candidate comparison
	xcomp = linspace(-6, 0, 601)';
	dxcomp = (xcomp(end) - xcomp(1)) / (length(xcomp) - 1);
	cweight = exp(-0.5*(xcomp/3).^2);  % Standard deviation of 3
	cweighttotal = dxcomp * (sum(cweight) - (cweight(1) + cweight(end))/2);

	% Plot all candidates if required
	if plot_all
		PlotCurves(71, candidates(1:2,:), f, xcomp, mish, mishgc, swish, swishgc, xminmish, fminmish, xminswish, fminswish, 'Default simplest'); pause;
		PlotCurves(71, candidates(3:4,:), f, xcomp, mish, mishgc, swish, swishgc, xminmish, fminmish, xminswish, fminswish, 'Minimum point matched'); pause;
		PlotCurves(71, candidates(5:8,:), f, xcomp, mish, mishgc, swish, swishgc, xminmish, fminmish, xminswish, fminswish, 'Curve matched'); pause;
		PlotCurves(71, candidates(9:12,:), f, xcomp, mish, mishgc, swish, swishgc, xminmish, fminmish, xminswish, fminswish, 'Derivative matched'); pause;
		for j = 13:beta_num:size(candidates,1)
			PlotCurves(71, candidates(j:j+beta_num-1,:), f, xcomp, mish, mishgc, swish, swishgc, xminmish, fminmish, xminswish, fminswish, sprintf('alpha = %.4f', candidates(j,1))); pause;
		end
	end

	% Select indices that will definitely be kept
	[~, I0] = min((candidates(:,1) - 1).^2 + (candidates(:,2) - 1).^2);  % Closest to (1,1)
	[~, I1] = min((candidates(:,1) - 2).^2 + (candidates(:,2) - 1).^2);  % Closest to (2,1)
	[~, IMA] = min(candidates(:,6));  % Least Gaussian-weighted difference to mish
	[~, ISA] = min(candidates(:,7));  % Least Gaussian-weighted difference to swish
	[~, IMM] = min((candidates(:,4) - xminmish).^2 + (candidates(:,5) - fminmish).^2);    % Closest minimum point to mish minimum point
	[~, ISM] = min((candidates(:,4) - xminswish).^2 + (candidates(:,5) - fminswish).^2);  % Closest minimum point to swish minimum point
	[~, IMDA] = min(candidates(:,10));  % Least Gaussian-weighted derivative difference to mish
	[~, ISDA] = min(candidates(:,11));  % Least Gaussian-weighted derivative difference to swish
	[~, IMGCA] = min(candidates(:,8));  % Least Gaussian-weighted difference to gradient-corrected mish
	[~, ISGCA] = min(candidates(:,9));  % Least Gaussian-weighted difference to gradient-corrected swish
	[~, IMDGCA] = min(candidates(:,12));  % Least Gaussian-weighted derivative difference to gradient-corrected mish
	[~, ISDGCA] = min(candidates(:,13));  % Least Gaussian-weighted derivative difference to gradient-corrected swish
	keep = [I0 I1 IMA ISA IMM ISM IMDA ISDA IMGCA ISGCA IMDGCA ISDGCA];

	% Select which candidates to keep
	chosen = nan(0, size(candidates, 2));
	[chosen] = FilterCandidates(candidates(keep,:), chosen, similarity_thres / 4, f, xcomp, dxcomp, cweight, cweighttotal, area_range, xmin_range, fmin_range);
	[chosen] = FilterCandidates(candidates, chosen, similarity_thres, f, xcomp, dxcomp, cweight, cweighttotal, area_range, xmin_range, fmin_range);  % Note: Randomisation via candidates(randperm(size(candidates,1)),:) does not overall seem to deliver an overwhelming diversity improvement

	% Plot the selected activation functions
	PlotCurves(72, chosen, f, xcomp, mish, mishgc, swish, swishgc, xminmish, fminmish, xminswish, fminswish);
	
	% Strip it down to just the parameters and incorporate gamma
	chosen = chosen(:,1:2);
	gamma_value = linspace(gamma_range(1), gamma_range(2), gamma_num);
	gamma_vals = repmat(gamma_value, size(chosen,1), 1);
	chosen = [repmat(chosen, gamma_num, 1) gamma_vals(:)];
	
end

% Find minimum point
function [xmin, fmin] = MinimumPoint(O, f, varargin)
	[xmin, fmin, info] = fmincon(@(x) f(x, varargin{:}), -0.1, [], [], [], [], -20.0, 0.0, [], O);
	if info <= 0
		warning('Failed to find minimum point');
	end
end

% Cost function for minimum point matching
function [cost] = PointMatchingCost(f, alpha, beta, xminref, fminref, O)
	[xmin, fmin] = MinimumPoint(O, f, alpha, beta);
	cost = (xmin - xminref).^2 + (fmin - fminref).^2;
end

% Find minimum point matching parameters
function [pmin] = MinimumPointMatchingCost(f, xminref, fminref, O)
	func = @(p) PointMatchingCost(f, p(1), p(2), xminref, fminref, O);
	[pmin, ~, info] = fminunc(func, [0.25 0.25], O);
	if info <= 0
		warning('Failed to find minimum point matching parameters');
	end
end

% Cost function for curve matching
function [cost] = CurveMatchingCost(f, xparam, alpha, beta, fparamref, dx, weight, weighttotal)
	fparam = f(xparam, alpha, beta);
	cost = CurveMatchingCostFP(fparam, fparamref, dx, weight, weighttotal);
end
function [cost] = CurveMatchingCostFP(fparam, fparamref, dx, weight, weighttotal)
	costdata = weight .* (fparam - fparamref).^2;
	cost = dx * (sum(costdata) - (costdata(1) + costdata(end))/2) / weighttotal;
end

% Find minimum curve matching parameters
function [pmin] = MinimumCurveMatching(f, xparam, fparamref, dx, weight, weighttotal, O)
	func = @(p) CurveMatchingCost(f, xparam, p(1), p(2), fparamref, dx, weight, weighttotal);
	[pmin, ~, info] = fmincon(func, [0.25 0.25], [], [], [], [], [1e-6 1e-6], [1e3 1e3], [], O);
	if info <= 0
		warning('Failed to find minimum curve matching parameters');
	end
end

% Cost function for derivative matching
function [cost] = DerivMatchingCost(f, xparam, alpha, beta, dfparamref, dx, dweight, dweighttotal)
	fparam = f(xparam, alpha, beta);
	dfparam = diff(fparam);
	cost = DerivMatchingCostFP(dfparam, dfparamref, dx, dweight, dweighttotal);
end
function [cost] = DerivMatchingCostFP(dfparam, dfparamref, dx, dweight, dweighttotal)
	costdata = dweight .* ((dfparam - dfparamref) / dx).^2;
	cost = dx * sum(costdata) / dweighttotal;
end

% Find minimum derivative matching parameters
function [pmin] = MinimumDerivMatching(f, xparam, dfparamref, dx, dweight, dweighttotal, O)
	func = @(p) DerivMatchingCost(f, xparam, p(1), p(2), dfparamref, dx, dweight, dweighttotal);
	[pmin, ~, info] = fmincon(func, [0.25 0.25], [], [], [], [], [1e-6 1e-6], [1e3 1e3], [], O);
	if info <= 0
		warning('Failed to find minimum curve matching parameters');
	end
end

% Plot curves
function PlotCurves(fig, chosen, f, xcomp, mish, mishgc, swish, swishgc, xminmish, fminmish, xminswish, fminswish, name)
	figure(fig);
	plot(xcomp, [mish(xcomp) mishgc(xcomp)], 'r--');
	hold on;
	plot(xcomp, [swish(xcomp) swishgc(xcomp)], 'b--');
	plot(xminmish, fminmish, 'rx');
	plot(xminswish, fminswish, 'bx');
	ax = gca;
	for k = 1:size(chosen,1)
		color = get(ax, 'ColorOrderIndex');
		plot(xcomp, f(xcomp, chosen(k,1), chosen(k,2)), '-');
		set(ax, 'ColorOrderIndex', color)
		plot(chosen(k,4), chosen(k,5), '.');
	end
	hold off;
	grid on;
	axis([min(xcomp) max(xcomp) -1.1 0.1]);
	if nargin >= 13 && ~isempty(name)
		title(name);
	end
end

% Filter candidates
function [chosen] = FilterCandidates(candidates, chosen, thres, f, xcomp, dxcomp, cweight, cweighttotal, area_range, xmin_range, fmin_range)
	for I = 1:size(candidates,1)
		candidate = candidates(I,:);
		if candidate(3) < area_range(1) || candidate(3) > area_range(2) || candidate(4) < xmin_range(1) || candidate(4) > xmin_range(2) || candidate(5) < fmin_range(1) || candidate(5) > fmin_range(2)
			continue
		end
		skip = false;
		for J = 1:size(chosen,1)
			existing = chosen(J,:);
			if candidate(1) == existing(1) && candidate(2) == existing(2)
				skip = true;
				break
			end
			diffdata = cweight .* (f(xcomp, candidate(1), candidate(2)) - f(xcomp, existing(1), existing(2))).^2;
			difftotal = sqrt(dxcomp * (sum(diffdata) - (diffdata(1) + diffdata(end))/2) / cweighttotal);
			if difftotal <= thres
				skip = true;
				break
			end
		end
		if ~skip
			chosen(end+1,:) = candidate;  %#ok<AGROW>
		end
	end
end
% EOF
