% Classification losses experimentations
%
% Figure 101: Extrema
%  - There should be extrema that correspond to minimums, so there are situations of equilibrium where the loss is happy and no feedback is needed (no gradients flow)
%  - The extrema probabilities should as flexibly as possible cover all situations where (with some safety margin) pT is highest (especially after capping)
%  - The extrema loss L should be constant (e.g. function of ep), and not a function of any free parameter
%  - AFTER CAPPING: All capped areas become extrema that should have the same loss value as the existing extrema losses
%
% Figure 102: Convexity
%  - Loss function should be convex wherever it will not get capped anyway (i.e. should have non-negative/zero 'convexity' values in the plots)
%
% Figure 103: pT ranges 0->1 and the remaining probability is split up evenly across all pf
%  - Loss should be convex (concave up)
%  - d/dxT should be generally negative and monotonic increasing (greater xT consistently leads to less up-adjustment of xT via gradients)
%  - All d/dxf should be generally positive and monotonic decreasing (greater xT consistently leads to less down-adjustment of xf via gradients)
%  - The single minimum point of the loss should correspond to the desired equilibrium situation (calculated as probabilities pT, pf)
%  - The minimum point should correspond to an xT value that is not particularly far from 0
%  - Loss should approach Inf via a linear asymptote as xT -> -Inf
%  - Loss should approach Inf via a linear asymptote or flatten to horizontal as xT -> Inf
%  - d/dxT should have a dynamic range of 1
%  - AFTER CAPPING: Loss should be horizontal / all d/dx should be exactly zero to the right of the minimum point
%
% Figure 104: pT ranges 0->1 and the remaining probability is all given to pf1
%  - Loss should be convex (concave up)
%  - d/dxT should be generally negative and monotonic increasing (greater xT consistently leads to less up-adjustment of xT via gradients)
%  - d/dxf1 should be generally positive and monotonic decreasing (greater xT consistently leads to less down-adjustment of xf via gradients)
%  - Other d/dxf's should be essentially zero, or ep-scale negative in order to pull the probabilities from 0 up to small finite values
%  - The single minimum point of the loss should correspond to the desired equilibrium situation (calculated as probabilities pT, pf1)
%  - The minimum point should correspond to an xT value that is not particularly far from 0
%  - Loss should approach Inf via a linear asymptote as xT -> -Inf
%  - Loss should approach Inf via a linear asymptote or flatten to horizontal as xT -> Inf
%  - d/dxT and d/dxf1 should have a dynamic range of 1
%  - AFTER CAPPING: Loss should be horizontal / all d/dx should be exactly zero to the right of the minimum point
%
% Figure 105/106: Loss should be essentially constant relative to xd (even if there are gradients flowing to all variables to e.g. reach a suitable pT)
%
% Figure 107: Loss should have a clear minimum at xd = 0
%

% Reset
clear;
clc;

% Symbolic variables
syms xT xf1 xf2 xf3 xf xd real;
syms ep real positive;
assume(0 < ep & ep < 1);

% Common terms
K = 4;
en = 1 - ep;
epp = ep / (K - 1);
eta = log(1-ep) - log(ep/(K-1));
tau = (1 - 1 / (K*(1 - ep)))^2;

% Softmax
pT = exp(xT) / (exp(xT) + exp(xf1) + exp(xf2) + exp(xf3));
pf1 = exp(xf1) / (exp(xT) + exp(xf1) + exp(xf2) + exp(xf3));
pf2 = exp(xf2) / (exp(xT) + exp(xf1) + exp(xf2) + exp(xf3));
pf3 = exp(xf3) / (exp(xT) + exp(xf1) + exp(xf2) + exp(xf3));
assert(isequal(simplify(pT + pf1 + pf2 + pf3), sym(1)));

% Loss terms (C constants scale each loss so that the gradient norm/square-norm is 1 when all X are equal)
CNLL = sqrt(K/(K-1));
CXNLL = sqrt((K-1)/K) / (1 - ep - 1/K);
CRRL = sqrt(K/(K-1)) / (2*eta);
CSRRL = 1 / sqrt(tau);
J = (xT-eta)^2 + xf1^2 + xf2^2 + xf3^2 - (xT-eta + xf1 + xf2 + xf3).^2 / K;
delta = (1-tau)/tau * (K-1)/K * eta^2;

% Define losses
losses = cell(0);
loss_names = cell(0);
[losses, loss_names] = define_loss(losses, loss_names, 'NLL', -CNLL * log(pT));
[losses, loss_names] = define_loss(losses, loss_names, 'KLDiv', CXNLL * (en*(log(en) - log(pT)) + epp*(log(epp) - log(pf1)) + epp*(log(epp) - log(pf2)) + epp*(log(epp) - log(pf3))));
[losses, loss_names] = define_loss(losses, loss_names, 'SNLL', -CXNLL * (en*log(pT) + epp*(log(pf1) + log(pf2) + log(pf3))));
[losses, loss_names] = define_loss(losses, loss_names, 'DNLL', -CXNLL * (en*log(pT) + ep*log(1-pT)));
[losses, loss_names] = define_loss(losses, loss_names, 'RRL', CRRL*J);
[losses, loss_names] = define_loss(losses, loss_names, 'SRRL', CSRRL*(sqrt(J + delta) - sqrt(delta)));
% TODO: MSE / Focal loss / others from wikipedia
% TODO: Loss that calculates desired target pT based on current distribution and capped dual-logs towards it (non-constant terms in front of logs)

% Plot variables
x = linspace(-10, 10, 2001)';
subplot_names = {'Loss', 'Probs', 'd/dx_T', 'd/dx_{f1}', 'd/dx_{f2}', 'dL/dt'};
epdbl = 0.2;
fig = 101;
S = 12;
subR = 2;
subC = ceil(numel(losses)/subR);

% Display loss functions
fprintf("LOSS FUNCTIONS\n\n");
for l = 1:numel(losses)
	fprintf("%s: \n", loss_names{l});
	pretty(losses{l});
end
	
% Calculate partial derivatives
fprintf("PARTIAL DERIVATIVES\n\n");
lossesdT = cell(size(losses));
lossesdf1 = cell(size(losses));
lossesdf2 = cell(size(losses));
lossesdf3 = cell(size(losses));
lossesdL = cell(size(losses));
for l = 1:numel(losses)
	lossesdT{l} = simplify(diff(losses{l}, xT));
	lossesdf1{l} = simplify(diff(losses{l}, xf1));
	lossesdf2{l} = simplify(diff(losses{l}, xf2));
	lossesdf3{l} = simplify(diff(losses{l}, xf3));
	lossesdL{l} = simplify(-(lossesdT{l}^2 + lossesdf1{l}^2 + lossesdf2{l}^2 + lossesdf3{l}^2));
	fprintf("%s d/dxT:\n", loss_names{l});
	pretty(lossesdT{l});
	fprintf("%s d/dxf1:\n", loss_names{l});
	pretty(lossesdf1{l});
	fprintf("%s d/dxf2:\n", loss_names{l});
	pretty(lossesdf2{l});
	fprintf("%s d/dxf3:\n", loss_names{l});
	pretty(lossesdf3{l});
	assert(isequal(simplify(lossesdT{l} + lossesdf1{l} + lossesdf2{l} + lossesdf3{l}), sym(0)));
end

% Extrema
fprintf("EXTREMA\n\n");
figure(fig);
fig = fig + 1;
sgtitle("Extrema");
for l = 1:numel(losses)
	subplot(subR, subC, l);
	extrema = solve([lossesdT{l} lossesdf1{l} lossesdf2{l} lossesdf3{l}], [xT xf1 xf2 xf3], 'ReturnConditions', true);
	fprintf("%s extrema conditions as func of %s:\n", loss_names{l}, strtrim(formattedDisplayText(extrema.parameters)));
	pretty(simplify(extrema.conditions));
	fprintf("%s extrema [xT xf1 xf2 xf3]:\n", loss_names{l});
	pretty([collect(simplify(extrema.xT)) collect(simplify(extrema.xf1)) collect(simplify(extrema.xf2)) collect(simplify(extrema.xf3))]);
	if ~isempty(extrema.xT)
		fprintf("%s extrema [pT pf1 pf2 pf3]:\n", loss_names{l});
		extremap = [collect(simplify(subs(pT, [xT xf1 xf2 xf3], [extrema.xT extrema.xf1 extrema.xf2 extrema.xf3]))) collect(simplify(subs(pf1, [xT xf1 xf2 xf3], [extrema.xT extrema.xf1 extrema.xf2 extrema.xf3]))) collect(simplify(subs(pf2, [xT xf1 xf2 xf3], [extrema.xT extrema.xf1 extrema.xf2 extrema.xf3]))) collect(simplify(subs(pf3, [xT xf1 xf2 xf3], [extrema.xT extrema.xf1 extrema.xf2 extrema.xf3])))];
		pretty(extremap);
		fprintf("%s extrema loss L:\n", loss_names{l});
		extremaL = collect(simplify(subs(losses{l}, [xT xf1 xf2 xf3], [extrema.xT extrema.xf1 extrema.xf2 extrema.xf3])));
		pretty(extremaL);
		Z = 1 + randn(1000, length(extrema.parameters));
		conditions = subs(extrema.conditions, ep, epdbl);
		conditions = logical(subs(conditions, sym2cell(extrema.parameters), num2cell(Z,1)));
		extremapdbl = double(subs(subs(extremap, ep, epdbl), sym2cell(extrema.parameters), num2cell(Z(conditions,:),1)));
		extremaLdbl = double(subs(subs(extremaL, ep, epdbl), sym2cell(extrema.parameters), num2cell(Z(conditions,:),1)));
		scatter3(extremapdbl(:,2), extremapdbl(:,3), extremapdbl(:,4), S, extremaLdbl, 'filled');
		caxis([min(extremaLdbl)-eps(max(abs(min(extremaLdbl)),1)) max(extremaLdbl)+eps(max(abs(max(extremaLdbl)),1))]);
		h = colorbar;
	end
	axis equal;
	axis vis3d;
	xlim([0 1]);
	ylim([0 1]);
	zlim([0 1]);
	xlabel('p_{f1}');
	ylabel('p_{f2}');
	zlabel('p_{f3}');
	title(loss_names{l});
	view([48 14]);
	grid on;
end
drawnow;

% Convexity
fprintf("CONVEXITY\n\n");
figure(fig);
fig = fig + 1;
sgtitle("Convexity");
for l = 1:numel(losses)
	subplot(subR, subC, l);
	H = collect(simplify(hessian(losses{l}, [xT xf1 xf2 xf3])));
	X = 2*randn(1,4,1000);
	Xcol = reshape(X,4,1000)';
	Hdbl = double(subs(H, {ep, xT, xf1, xf2, xf3}, {epdbl, X(1,1,:), X(1,2,:), X(1,3,:), X(1,4,:)}));
	convexity = nan(size(Hdbl,3),1);
	for k = 1:size(Hdbl,3)
		convexity(k,1) = eigs(Hdbl(:,:,k), 1, 'smallestreal');
	end
	scatter3(double(subs(pf1, {ep, xT, xf1, xf2, xf3}, {epdbl, Xcol(:,1), Xcol(:,2), Xcol(:,3), Xcol(:,4)})), double(subs(pf2, {ep, xT, xf1, xf2, xf3}, {epdbl, Xcol(:,1), Xcol(:,2), Xcol(:,3), Xcol(:,4)})), double(subs(pf3, {ep, xT, xf1, xf2, xf3}, {epdbl, Xcol(:,1), Xcol(:,2), Xcol(:,3), Xcol(:,4)})), S, convexity, 'filled');
	colorbar;
	axis equal;
	axis vis3d;
	xlim([0 1]);
	ylim([0 1]);
	zlim([0 1]);
	xlabel('p_{f1}');
	ylabel('p_{f2}');
	zlabel('p_{f3}');
	title(loss_names{l});
	view([135 32]);
	grid on;
end
drawnow;

% Situation variables
loss_situation_args = {losses, lossesdT, lossesdf1, lossesdf2, lossesdf3, lossesdL, loss_names, ep, epdbl, subplot_names, x, xT, xf1, xf2, xf3, pT, pf1, pf2, pf3};

% Situations
fig = loss_situation(fig, "All xf are equal and WLOG let their value be zero", 'x_T', @(fn) collect(simplify(subs(fn, [xf1 xf2 xf3], [0 0 0]))), loss_situation_args{:});
fig = loss_situation(fig, "Effective 2-way classification xT vs xf1", 'x_T', @(fn) collect(simplify(limit(subs(fn, [xf1 xf2 xf3], [0 xf xf]), xf, -Inf))), loss_situation_args{:});
fig = loss_situation(fig, "pT = 1-ep and the rest divided amongst xf1 and equal xf2/3", 'x_d = x_{f1} - x_{f2/3}', @(fn) collect(simplify(subs(fn, [xT xf1 xf2 xf3], [0 log((1/(1-ep) - 1)/(1+2*exp(-xd))) repmat(log((1/(1-ep) - 1)/(exp(xd)+2)), 1, 2)]))), loss_situation_args{:});
fig = loss_situation(fig, "pT = 0.9-ep and the rest divided amongst xf1 and equal xf2/3", 'x_d = x_{f1} - x_{f2/3}', @(fn) collect(simplify(subs(fn, [xT xf1 xf2 xf3], [0 log((1/(0.9-ep) - 1)/(1+2*exp(-xd))) repmat(log((1/(0.9-ep) - 1)/(exp(xd)+2)), 1, 2)]))), loss_situation_args{:});
fig = loss_situation(fig, "pT = 1/K and the rest divided amongst xf1 and equal xf2/3", 'x_d = x_{f1} - x_{f2/3}', @(fn) collect(simplify(subs(fn, [xT xf1 xf2 xf3], [0 log((K - 1)/(1+2*exp(-xd))) repmat(log((K - 1)/(exp(xd)+2)), 1, 2)]))), loss_situation_args{:});

%
% Functions
%

% Loss definition function
function [losses, loss_names] = define_loss(losses, loss_names, name, loss)
	losses{end+1} = loss;
	loss_names{end+1} = name;
end

% Situation function
function [fig] = loss_situation(fig, situation, depvar, loss_tfrm, losses, lossesdT, lossesdf1, lossesdf2, lossesdf3, lossesdL, loss_names, ep, epdbl, subplot_names, x, xT, xf1, xf2, xf3, pT, pf1, pf2, pf3)
	
	fprintf("SITUATION %d: %s\n\n", fig, situation);
	
	fprintf("LOGITS:\n");
	pretty(loss_tfrm([xT xf1 xf2 xf3]));
	fprintf("PROBABILITIES:\n");
	pretty(loss_tfrm([pT pf1 pf2 pf3]));
	
	figure(fig); clf;
	fig = fig + 1;
	
	subplot(2, 3, 2);
	hold on;
	plot(x, double(subs(subs(loss_tfrm(pT), ep, epdbl), x)), '-');
	plot(x, double(subs(subs(loss_tfrm(pf1), ep, epdbl), x)), '-');
	plot(x, double(subs(subs(loss_tfrm(pf2), ep, epdbl), x)), '-');
	plot(x, double(subs(subs(loss_tfrm(pf3), ep, epdbl), x)), '-');
	hold off;
	title(subplot_names{2});
	xlabel(depvar);
	legend('p_T', 'p_{f1}', 'p_{f2}', 'p_{f3}', 'Location', 'Best');
	grid on;
	
	for p = [1 3 4 5 6]
		subplot(2, 3, p);
		hold on;
	end
	
	for l = 1:numel(losses)
		loss = loss_tfrm(losses{l});
		lossdT = loss_tfrm(lossesdT{l});
		lossdf1 = loss_tfrm(lossesdf1{l});
		lossdf2 = loss_tfrm(lossesdf2{l});
		lossdf3 = loss_tfrm(lossesdf3{l});
		lossdL = loss_tfrm(lossesdL{l});
		fprintf("%s:\n", loss_names{l});
		pretty(loss);
		fprintf("%s d/dxT:\n", loss_names{l});
		pretty(lossdT);
		fprintf("%s d/dxf1:\n", loss_names{l});
		pretty(lossdf1);
		fprintf("%s d/dxf2:\n", loss_names{l});
		pretty(lossdf2);
		fprintf("%s d/dxf3:\n", loss_names{l});
		pretty(lossdf3);
		subplot(2, 3, 1);
		plot(x, double(subs(subs(loss, ep, epdbl), x)), '-');
		subplot(2, 3, 3);
		plot(x, double(subs(subs(lossdT, ep, epdbl), x)), '-');
		subplot(2, 3, 4);
		plot(x, double(subs(subs(lossdf1, ep, epdbl), x)), '-');
		subplot(2, 3, 5);
		plot(x, double(subs(subs(lossdf2, ep, epdbl), x)), '-');
		subplot(2, 3, 6);
		plot(x, double(subs(subs(lossdL, ep, epdbl), x)), '-');
	end
	
	for p = [1 3 4 5 6]
		subplot(2, 3, p);
		title(subplot_names{p});
		xlabel(depvar);
		legend(loss_names{:}, 'Location', 'Best');
		hold off;
		grid on;
	end
	
	sgtitle(situation);
	drawnow;
	
end
% EOF
