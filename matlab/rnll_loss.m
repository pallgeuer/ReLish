% Playground for numerical verification of RNLL loss candidates
% SITUATION: Specify pT, pF, all remaining probability is split evenly among remaining K-2 (and xT = 0)

% Main function
function rnll_loss(pTdes, pFdes, K, fig)

	% Clear console
	clc;

	% Default arguments
	if nargin < 1
		pTdes = 0.25;
	end
	if nargin < 2
		pFdes = 0.5;
	end
	if nargin < 3
		K = 6;
	end
	if nargin < 4
		fig = 201;
	end
	assert(K >= 3 && pTdes >= 0 && pFdes >= 0 && pTdes + pFdes <= 1);
	
	% Symbolic variables
	syms xT xF real;
	syms xK [1 K-2] real;

	% Common terms
	ep = 0.2;
	en = 1 - ep;                     % Equilibrium pT (if applicable)
	epp = ep / (K - 1);              % Equilibrium pF (if applicable)
	eta = log(1-ep) - log(ep/(K-1)); % Eta such that X = (eta, 0, ...) corresponds to P = (1-ep, ep/(K-1), ...)
	tau = (1 - 1 / (K*(1 - ep)))^2;  % Initial experimentation suggests 0.6-1.0x this expression (Ratio of all-X-equal gradient squared-norm to the max saturated value thereof)
	
	% Calculate logits and probabilities
	X = [0 log(pFdes)-log(pTdes) repmat(log((1-pFdes-pTdes)/(K-2))-log(pTdes),1,K-2)];
	Z = X(2:end) + eta;
	P = exp(X) ./ sum(exp(X));
	assert(abs(P(1) - pTdes) < 16*eps && abs(P(2) - pFdes) < 16*eps);
	
	% Softmax
	xFK = [xF xK];
	softmax_denom = exp(xT) + sum(exp(xFK));
	pT = exp(xT) / softmax_denom;
	pF = exp(xF) / softmax_denom;
	pK = exp(xK) / softmax_denom;
	pFK = [pF pK];

	% Loss terms (C constants scale each loss so that the gradient norm/square-norm is 1 when all X are equal)
	CNLL = sqrt(K/(K-1));
	CXNLL = sqrt((K-1)/K) / (1 - ep - 1/K);
	CRRL = sqrt(K/(K-1)) / (2*eta);
	CSRRL = 1 / sqrt(tau);
	J = (xT-eta).^2 + sum(xFK.^2) - (xT-eta + sum(xFK)).^2 / K;
	delta = (1-tau)/tau * (K-1)/K * eta^2;
	
	% Define losses
	losses = cell(0);
	loss_names = cell(0);
	[losses, loss_names] = define_loss(losses, loss_names, 'NLL', -CNLL * log(pT));  % Negative log likelihood loss
	[losses, loss_names] = define_loss(losses, loss_names, 'KLDiv', CXNLL * (en*(log(en) - log(pT)) + epp*(log(epp) - log(pF)) + sum(epp*(log(epp) - log(pK)))));  % Kullback-Leibler divergence loss: Identical to SNLL up to constant shift
	[losses, loss_names] = define_loss(losses, loss_names, 'SNLL', -CXNLL * (en*log(pT) + epp*sum(log(pFK))));  % Label-smoothed negative log likelihood loss
	[losses, loss_names] = define_loss(losses, loss_names, 'DNLL', -CXNLL * (en*log(pT) + ep*log(1-pT)));  % Dual negative log likelihood loss
	[losses, loss_names] = define_loss(losses, loss_names, 'RRL', CRRL*J);  % Relative raw logit loss
	[losses, loss_names] = define_loss(losses, loss_names, 'SRRL', CSRRL*sqrt(J + delta));  % Saturated raw logit loss
	
	% Plot constants
	green = [0.4660 0.6740 0.1880];
	red = [0.8500 0.3250 0.0980];
	labels = [{'T', 'F'} repmat({'K'},1,K-2)];
	
	% Plot logits and probabilities
	figure(fig);
	fig = fig + 1;
	ax = subplot(1,2,1);
	h = bar(X, 'FaceColor', 'flat');
	h.CData(1:2,:) = [green; red];
	set(ax, 'XTickLabel', labels);
	title('x_{*}');
	grid on;
	ax = subplot(1,2,2);
	h = bar(P, 'FaceColor', 'flat');
	h.CData(1:2,:) = [green; red];
	set(ax, 'XTickLabel', labels);
	title('p_{*}');
	grid on;
	drawnow;
	
	% Display loss functions
	fprintf("LOSSES\n\n");
	for l = 1:numel(losses)
		fprintf("%s: \n", loss_names{l});
		pretty(losses{l});
	end
	
	% Calculate gradients
	fprintf("GRADIENTS\n\n");
	lossesdx = cell(size(losses));
	for l = 1:numel(losses)
		lossesdx{l} = simplify(gradient(losses{l}, [xT xF xK]));
		fprintf("%s grad:\n", loss_names{l});
		pretty(lossesdx{l});
		assert(isequal(simplify(sum(lossesdx{l})), sym(0)));
	end
	
	% Calculate update rates
	fprintf("UPDATE RATES\n\n");
	dLdx = cat(2,lossesdx{:});
	dxdt = -dLdx;  % dx/dt = -dL/dx is the desired update rate of the raw logits (weights will be changed at a rate to try to perform this, up to learning rate and optimizer complexities)
	dzdt = dxdt(2:end,:) - dxdt(1,:);
	
	% Plot update rates (relative logits)
	figure(fig);
	fig = fig + 1;
	ax = subplot(1,2,1);
	h = bar(Z, 'FaceColor', 'flat');
	h.CData(1,:) = red;
	set(ax, 'XTickLabel', labels(2:end));
	title('z_{*}');
	grid on;
	ax = subplot(1,2,2);
	h = bar(double(subs(dzdt, [xT xF xK], X)), 1.0);
	loss_colors = cat(1, h.FaceColor);
	set(ax, 'XTickLabel', labels(2:end));
	legend(loss_names{:}, 'Location', 'Best');
	title('dz/dt');
	grid on;
	drawnow;
	
	% Calculate derived update rates
	fprintf("DERIVED UPDATE RATES\n\n");
	dpdt = jacobian([pT pF pK], [xT xF xK]) * dxdt;
	dLdt = dot(dLdx, dxdt, 1);
	
	% Plot update rates (logits/probabilities)
	figure(fig);
	fig = fig + 1;
	ax = subplot(1,2,1);
	bar(double(subs(dxdt, [xT xF xK], X)), 1.0);
	set(ax, 'XTickLabel', labels);
	legend(loss_names{:}, 'Location', 'Best');
	title('dx/dt');
	grid on;
	ax = subplot(1,2,2);
	bar(double(subs(dpdt, [xT xF xK], X)), 1.0);
	set(ax, 'XTickLabel', labels);
	legend(loss_names{:}, 'Location', 'Best');
	title('dp/dt');
	grid on;
	drawnow;
	
	% Plot update rates (loss)
	figure(fig);
	fig = fig + 1; %#ok<NASGU>
	ax = subplot(1,1,1);
	bar(double(subs(dLdt, [xT xF xK], X)), 'FaceColor', 'flat', 'CData', loss_colors);
	set(ax, 'XTickLabel', loss_names);
	title('dL/dt');
	grid on;
	drawnow;

end

% Loss definition function
function [losses, loss_names] = define_loss(losses, loss_names, name, loss)
	losses{end+1} = loss;
	loss_names{end+1} = name;
end
% EOF
