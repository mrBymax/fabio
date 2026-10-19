% =========================================================================
% fabio - Auction Simulator & Scenario Engine (Stage 1 Extension)
% =========================================================================
% This script simulates different auction scenarios (What-If Analysis)
% and runs Monte Carlo simulations with price volatility and opponent snatches
% to find the most resilient targets for your draft.
% =========================================================================

clear; clc;

% 1. Load Dataset
dataPath = fullfile(fileparts(mfilename("fullpath")), '..', 'dataset', 'quotazioni_fantacalcio.csv');
if ~isfile(dataPath)
    dataPath = fullfile(fileparts(mfilename("fullpath")), '..', 'dataset', 'demo_quotazioni.csv');
end
data = readtable(dataPath, 'TextType', 'string');

budget = 500;
data.EstPrice = max(1, round(double(data.FVM) / 1000 * budget));

fprintf('=================================================================\n');
fprintf('       fabio - AUCTION SCENARIO & MONTE CARLO SIMULATOR     \n');
fprintf('=================================================================\n\n');

%% =========================================================================
% PART 1: "WHAT-IF" SCENARIO ANALYSIS
% =========================================================================
fprintf('--- PART 1: SCENARIO COMPARISONS ---\n');

% Define 3 different draft strategy scenarios
scenarios = struct();

% Scenario A: Mega-Top Striker Strategy (Lock Lautaro)
scenarios(1).Name     = "Scenario A: Mega-Top Striker (Lautaro Martinez)";
scenarios(1).Locked   = ["Martinez L."];
scenarios(1).Excluded = [];

% Scenario B: Heavy Midfield Spine (Calhanoglu + McTominay + Paz, No Lautaro)
scenarios(2).Name     = "Scenario B: Heavy Midfield Spine";
scenarios(2).Locked   = ["Calhanoglu", "McTominay", "Paz N."];
scenarios(2).Excluded = ["Martinez L.", "Malen"];

% Scenario C: Dual Tier-2 Strikers (Thuram + Kean)
scenarios(3).Name     = "Scenario C: Dual Tier-2 Strikers (Thuram + Kean)";
scenarios(3).Locked   = ["Thuram", "Kean"];
scenarios(3).Excluded = ["Martinez L."];

for s = 1:length(scenarios)
    fprintf('\n>>> Running %s...\n', scenarios(s).Name);
    [squad, spent, fvm, ok] = run_single_optimization(data, budget, ...
        scenarios(s).Locked, scenarios(s).Excluded, 0);
    
    if ok
        fprintf('    Total Spent: %d / %d crediti | Total FVM: %d\n', spent, budget, fvm);
        fprintf('    Key Starters Selected:\n');
        
        top_fwd = squad(squad.R == "A", :);
        top_fwd = sortrows(top_fwd, 'FVM', 'descend');
        fprintf('      [FWD]: %s\n', strjoin(top_fwd.Nome(1:min(3, height(top_fwd))), ', '));
        
        top_mid = squad(squad.R == "C", :);
        top_mid = sortrows(top_mid, 'FVM', 'descend');
        fprintf('      [MID]: %s\n', strjoin(top_mid.Nome(1:min(3, height(top_mid))), ', '));
        
        top_def = squad(squad.R == "D", :);
        top_def = sortrows(top_def, 'FVM', 'descend');
        fprintf('      [DEF]: %s\n', strjoin(top_def.Nome(1:min(2, height(top_def))), ', '));
    else
        fprintf('    No feasible solution found for this scenario.\n');
    end
end


%% =========================================================================
% PART 2: MONTE CARLO AUCTION SIMULATION (50 Iterations)
% =========================================================================
fprintf('\n\n=================================================================\n');
fprintf('--- PART 2: MONTE CARLO SIMULATION (50 DRAFT RUNS) ---\n');
fprintf('Simulating price volatility (+/- 15%%) & rival snatches...\n');

num_sims = 50;
pick_counts = zeros(height(data), 1);
successful_runs = 0;

for sim = 1:num_sims
    % 1. Randomly simulate rivals snatching 6-10 top players
    top_player_indices = find(data.FVM > 150);
    num_snatches = randi([6, 10]);
    snatched_idx = randsample(top_player_indices, min(num_snatches, length(top_player_indices)));
    excluded_names = data.Nome(snatched_idx);
    
    % 2. Run optimization with 12% price noise
    price_noise_std = 0.12;
    [squad, ~, ~, ok] = run_single_optimization(data, budget, [], excluded_names, price_noise_std);
    
    if ok
        successful_runs = successful_runs + 1;
        % Track selected player IDs/indices
        [~, loc] = ismember(squad.Id, data.Id);
        pick_counts(loc) = pick_counts(loc) + 1;
    end
end

fprintf('Completed %d / %d successful simulated auctions.\n\n', successful_runs, num_sims);

% Display Top Draft Frequency Targets (Must-Buys across simulations)
data.PickRate = (pick_counts / successful_runs) * 100;

roles = ["P", "D", "C", "A"];
role_titles = ["🧤 TOP GOALKEEPERS", "🛡️ TOP DEFENDERS", "⚙️ TOP MIDFIELDERS", "⚡ TOP FORWARDS"];

for r = 1:length(roles)
    sub = data(data.R == roles(r) & data.PickRate > 20, {'Nome', 'Squadra', 'EstPrice', 'FVM', 'PickRate'});
    sub = sortrows(sub, 'PickRate', 'descend');
    
    fprintf('================ %s (Pick Rate > 20%%) ================\n', role_titles(r));
    if isempty(sub)
        fprintf('  (No single player exceeded 20%% frequency in this department)\n');
    else
        disp(sub(1:min(8, height(sub)), :));
    end
end


%% =========================================================================
% HELPER FUNCTION: Single MILP Squad Optimizer
% =========================================================================
function [squad, total_spent, total_fvm, success] = run_single_optimization( ...
    data, budget, locked_names, excluded_names, noise_std)

    N = height(data);
    intcon = 1:N;
    lb = zeros(N, 1);
    ub = ones(N, 1);

    % Apply price volatility noise if requested
    if noise_std > 0
        noise = 1 + randn(N, 1) * noise_std;
        sim_prices = max(1, round(data.EstPrice .* noise));
    else
        sim_prices = data.EstPrice;
    end

    % Objective: Maximize FVM
    f = -double(data.FVM);

    % Role masks
    isGK  = data.R == "P";
    isDEF = data.R == "D";
    isMID = data.R == "C";
    isFWD = data.R == "A";

    isTopGK = isGK & (data.FVM >= 50);
    isMidGK = isGK & (data.FVM >= 10 & data.FVM < 50);
    isLowGK = isGK & (data.FVM < 10);

    % Equality Constraints: Exact Roster Quotas
    Aeq = [
        double(isTopGK');
        double(isMidGK');
        double(isLowGK');
        double(isDEF');
        double(isMID');
        double(isFWD')
    ];
    beq = [1; 1; 1; 8; 8; 6];

    % Inequality Constraints: Budget Caps
    A_total   = sim_prices';
    b_total   = budget;
    A_gk_max  = (sim_prices .* isGK)';
    A_def_max = (sim_prices .* isDEF)';
    A_mid_max = (sim_prices .* isMID)';
    A_fwd_max = (sim_prices .* isFWD)';

    A = [A_total; A_gk_max; A_def_max; A_mid_max; A_fwd_max];
    b = [b_total; 45; 70; 140; 320];

    % Lock selected players (lb = ub = 1)
    if ~isempty(locked_names)
        [is_locked, ~] = ismember(data.Nome, locked_names);
        if any(is_locked)
            lb(is_locked) = 1;
            ub(is_locked) = 1;
        end
    end

    % Exclude unavailable/opponent players (lb = ub = 0)
    if ~isempty(excluded_names)
        [is_excl, ~] = ismember(data.Nome, excluded_names);
        if any(is_excl)
            lb(is_excl) = 0;
            ub(is_excl) = 0;
        end
    end

    % Solve
    options = optimoptions('intlinprog', 'Display', 'off');
    [x_opt, ~, exitflag] = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub, options);

    if exitflag == 1
        success = true;
        squad = data(x_opt > 0.5, :);
        squad.SimPrice = sim_prices(x_opt > 0.5);
        total_spent = sum(squad.SimPrice);
        total_fvm = sum(squad.FVM);
    else
        success = false;
        squad = table();
        total_spent = 0;
        total_fvm = 0;
    end
end
