% Load the dataset
path = fullfile(fileparts(mfilename("fullpath")), '..', 'dataset','quotazioni_fantacalcio.csv');
data = readtable(path, 'TextType','string');

% Parameters
N = height(data); % number of players
budget = 500;
intcon = 1:N; % all decision variables are integers
lb = zeros(N,1); % x_i >= 0
ub = ones(N, 1); % x_i <= 1

f = -double(data.FVM); % we want to maximise value (FantaValoreMercato - FVM) so we minimise -FVM

% Budget Inequalities
A = double(data.Qt_A)'; % 1 x N row vector
b = budget;

% Roles quota:
isGK = data.R == "P";
isDEF = data.R == "D";
isMID = data.R == "C";
isFWD = data.R == "A";

Aeq = [
    double(isGK');
    double(isDEF');
    double(isMID');
    double(isFWD')
    ];

beq = [3;8;8;6]; % following 'Lega' rules


% Solve & Display Output
options = optimoptions('intlinprog', 'Display', 'final');
[x_opt, fval, exitflag] = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub, options);

if exitflag == 1
    selected = data(x_opt > 0.5, :);
    fprintf('\n=== OPTIMAL 25-PLAYER SQUAD ===\n');
    fprintf('Total Players: %d\n', height(selected));
    fprintf('Total Spent:   %d / %d crediti\n', sum(selected.Qt_A), budget);
    fprintf('Total FVM:     %d\n\n', sum(selected.FVM));

    % Display by role (sorted as: P,D,C,A)
    roles = ["P", "D", "C", "A"];
    role_titles = [
        "GOALKEEPERS (3)"
        "DEFENDERS (8)"
        "MIDFIELDERS (8)"
        "FORWARDS (6)"
    ];

    for r = 1:length(roles)
        fprintf('\n================ %s ================\n', role_titles(r));
        sub = selected(selected.R == roles(r), {'Nome', 'Squadra', 'Qt_A', 'FVM'});
        % Sort by FVM descending
        sub = sortrows(sub, 'FVM', 'descend');
        disp(sub);
    end
else
    fprintf('Solver did not find an optimal solution. Exitflag: %d\n', exitflag);
end