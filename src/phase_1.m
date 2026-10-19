clear; clc;

% Load the dataset
path = fullfile(fileparts(mfilename("fullpath")), '..', 'dataset', 'quotazioni_fantacalcio.csv');
if ~isfile(path)
    path = fullfile(fileparts(mfilename("fullpath")), '..', 'dataset', 'demo_quotazioni.csv');
end
data = readtable(path, 'TextType','string');

% 2. Problem Parameters
N = height(data);       % Number of total players
budget = 500;           % Total auction budget
intcon = 1:N;           % All decision variables are binary integers
lb = zeros(N, 1);       % x_i >= 0
ub = ones(N, 1);        % x_i <= 1

% 3. Auction Settings & Expected Prices (scaled from FVM base 1000)
estimated_price = max(1, round(double(data.FVM) / 1000 * budget));
data.EstPrice = estimated_price;

% 4. Objective: Maximize total FVM (minimize -FVM)
f = -double(data.FVM);

% 5. Role & Goalkeeper Tier Indicator Vectors
isGK  = data.R == "P";
isDEF = data.R == "D";
isMID = data.R == "C";
isFWD = data.R == "A";

% Goalkeeper Tiers: 1 Top, 1 Mid/Second, 1 Low-cost reserve
isTopGK = isGK & (data.FVM >= 50);
isMidGK = isGK & (data.FVM >= 10 & data.FVM < 50);
isLowGK = isGK & (data.FVM < 10);

% 6. Linear Equality Constraints (Exact Roster & Tier Quotas: Aeq * x = beq)
Aeq = [
    double(isTopGK');   % Exactly 1 Top GK
    double(isMidGK');   % Exactly 1 Mid GK
    double(isLowGK');   % Exactly 1 Low-cost reserve GK
    double(isDEF');     % Exactly 8 Defenders
    double(isMID');     % Exactly 8 Midfielders
    double(isFWD')      % Exactly 6 Forwards
    ];
beq = [1; 1; 1; 8; 8; 6]; % Total = 25 players

% 7. Linear Inequality Constraints (Budget Limits: A * x <= b)
% Total Budget inequality
A_total = estimated_price';
b_total = budget;

% Department Budget caps
A_gk_max  = (estimated_price .* isGK)';
A_def_max = (estimated_price .* isDEF)';
A_mid_max = (estimated_price .* isMID)';
A_fwd_max = (estimated_price .* isFWD)';

b_gk_max  = 45;   % Max 45 crediti for GKs
b_def_max = 70;   % Max 70 crediti for DEFs
b_mid_max = 140;  % Max 140 crediti for MIDs
b_fwd_max = 320;  % Max 320 crediti for FWDs

A = [
    A_total;
    A_gk_max;
    A_def_max;
    A_mid_max;
    A_fwd_max
    ];

b = [
    b_total;
    b_gk_max;
    b_def_max;
    b_mid_max;
    b_fwd_max
    ];

% 8. Solve MILP with intlinprog
options = optimoptions('intlinprog', 'Display', 'final');
[x_opt, fval, exitflag] = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub, options);

% 9. Display Optimized Squad
if exitflag == 1
    selected = data(x_opt > 0.5, :);
    fprintf('\n==================================================\n');
    fprintf('           F.A.B.I.O. OPTIMAL 25-PLAYER SQUAD      \n');
    fprintf('==================================================\n');
    fprintf('Total Players: %d\n', height(selected));
    fprintf('Total Spent:   %d / %d crediti (Remaining: %d)\n', ...
        sum(selected.EstPrice), budget, budget - sum(selected.EstPrice));
    fprintf('Total FVM:     %d\n\n', sum(selected.FVM));

    roles = ["P", "D", "C", "A"];
    role_titles = [
        "🧤 GOALKEEPERS (1 Top, 1 Mid, 1 Reserve)"
        "🛡️ DEFENDERS (8)"
        "⚙️ MIDFIELDERS (8)"
        "⚡ FORWARDS (6)"
        ];

    for r = 1:length(roles)
        sub = selected(selected.R == roles(r), {'Nome', 'Squadra', 'Qt_A', 'EstPrice', 'FVM'});
        sub = sortrows(sub, 'FVM', 'descend');

        dept_spent = sum(sub.EstPrice);
        fprintf('\n================ %s (Spent: %d crediti) ================\n', role_titles(r), dept_spent);
        disp(sub);
    end

    fprintf('\n----------------------------------------------\n');
    fprintf('TOTAL ESTIMATED AUCTION SPENT: %d / %d crediti\n', sum(selected.EstPrice), budget);
    fprintf('TOTAL SQUAD FVM: %d\n', sum(selected.FVM));
    fprintf('----------------------------------------------\n');
else
    fprintf('Solver did not find an optimal solution. Exitflag: %d\n', exitflag);
end