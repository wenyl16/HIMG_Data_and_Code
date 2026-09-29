clc;
clear;
close all;

%% ================== System parameter settings ==================
% Battery parameters
SOC_init = 0.713;      % Initial state of charge: 70%
SOC_min = 0.1;       % Normal operating lower limit: 10%
SOC_stop = 0.1;      % Shutdown threshold: 10%
SOC_max = 0.9;       % Maximum charging limit: 90%

% Efficiency parameters
eta_ch = 0.95;         % Charging efficiency
eta_dis = 0.95;        % Discharging efficiency

% Time parameters
dt = 0.5;             % Time step: 0.5 hours (30 minutes)
total_points = 336;   % Total data points (7 days x 48 points/day)
hours_sim = total_points * dt;  % Simulation duration: 168 hours (7 days)

%% ================== Read data ==================
% Read seven days of data
data = xlsread('data_7day.xlsx');

% Extract data (column 1: PV power; column 2: load power)
P_pv_raw = data(:,1);      % Raw PV power (kW)
P_load_raw = data(:,2);    % Raw load power (kW)

% Check data completeness
fprintf('========== Data information ==========\n');
fprintf('Original number of data points: %d\n', length(P_pv_raw));
fprintf('Data time step: %.1f hours\n', dt);
fprintf('Data duration: %.1f hours (%.1f days)\n', length(P_pv_raw)*dt, length(P_pv_raw)*dt/24);
fprintf('PV power range: %.2f - %.2f kW\n', min(P_pv_raw), max(P_pv_raw));
fprintf('Load power range: %.2f - %.2f kW\n', min(P_load_raw), max(P_load_raw));
fprintf('===============================\n\n');

% Ensure enough data for 168 hours
if length(P_pv_raw) < total_points
    warning('Insufficient data: %.0f points available; %d points required', length(P_pv_raw), total_points);
    total_points = length(P_pv_raw);
    hours_sim = total_points * dt;
end

% Retain only the required data points
P_pv_raw = P_pv_raw(1:total_points);
P_load_raw = P_load_raw(1:total_points);

% Create a half-hourly time series
t = (0:total_points-1) * dt;  % Time points (hours)

fprintf('Data points used: %d\n', total_points);
fprintf('Simulation duration: %.1f hours (%.1f days)\n', hours_sim, hours_sim/24);

%% ================== System capacity configuration ==================
% Configure system capacities using the observed data
P_pv_rated = max(P_pv_raw);     % Rated PV power (kW)
% P_bat_max = 3000;                % Maximum battery power (kW), user specified
% E_bat = 6000;                    % Battery capacity (kWh), user specified
P_bat_max = 5800;                % Maximum battery power (kW), user specified
E_bat = 11600;                    % Battery capacity (kWh), user specified

fprintf('\n========== System configuration ==========\n');
fprintf('Peak PV power: %.2f kW\n', max(P_pv_raw));
fprintf('Mean PV power: %.2f kW\n', mean(P_pv_raw));
fprintf('Rated PV power: %.2f kW\n', P_pv_rated);
fprintf('Rated battery capacity: %.2f kWh\n', E_bat);
fprintf('Maximum battery power: %.2f kW\n', P_bat_max);
fprintf('Initial SOC: %.1f%%\n', SOC_init*100);
fprintf('==============================\n\n');

%% ================== Main simulation loop ==================
% Initialise
SOC = SOC_init;                  % Initial SOC
P_pv_actual = zeros(total_points, 1);     % Actual PV output
P_load_actual = zeros(total_points, 1);   % Actual load
P_bat_actual = zeros(total_points, 1);    % Actual battery power (positive = charge, negative = discharge)
SOC_record = zeros(total_points, 1);      % SOC record
SOC_record(1) = SOC;

% State variables
is_running = true;               % System operating status
cutoff_time = hours_sim;         % Outage time (initially the full simulation duration)
cutoff_index = total_points;     % Outage time index
energy_deficit = 0;              % Accumulated energy deficit

% Start simulation
fprintf('Starting simulation (duration: %.1f hours, %d time points)...\n', hours_sim, total_points);
tic;

for k = 1:total_points
    if ~is_running
        break;  % System has shut down; stop the simulation
    end
    
    % Current power values, taken directly from raw data without scaling
    P_pv = P_pv_raw(k);    % Actual PV power (kW)
    P_load = P_load_raw(k); % Actual load power (kW)
    
    % Store actual values
    P_pv_actual(k) = P_pv;
    P_load_actual(k) = P_load;
    
    %% ===== Calculate available discharge capacity =====
    if SOC > SOC_min
        % Normal operation: available energy extends from current SOC to the lower limit
        E_dis_max = (SOC - SOC_min) * E_bat * eta_dis;
    else
        % Emergency operation: available energy extends from current SOC to the shutdown threshold
        E_dis_max = (SOC - SOC_stop) * E_bat * eta_dis;
    end
    
    % Maximum discharge power, limited by battery power and available energy
    P_bat_available = min(P_bat_max, E_dis_max / dt);
    
    %% ===== Check shutdown conditions =====
    if (P_pv + P_bat_available < P_load) || (SOC <= SOC_stop)
        is_running = false;
        cutoff_time = t(k);
        cutoff_index = k;
        energy_deficit = (P_load - P_pv - P_bat_available) * dt;
        fprintf('\nSystem shut down at %.2f hours (%.2f days)\n', cutoff_time, cutoff_time/24);
        fprintf('At shutdown: time point=%d, PV=%.2f kW, load=%.2f kW, SOC=%.2f%%\n', ...
                k, P_pv, P_load, SOC*100);
        break;
    end
    
    %% ===== Power balance and battery charging/discharging strategy =====
    P_diff = P_pv - P_load;  % Power difference
    
    if P_diff >= 0
        % PV surplus: charge the battery
        % Limit charging power to the battery power rating
        P_ch = min(P_diff, P_bat_max);
        
        % Limit charging power by the available battery capacity
        E_ch_max = (SOC_max - SOC) * E_bat / eta_ch;
        P_ch = min(P_ch, E_ch_max / dt);
        
        % Update SOC
        SOC = SOC + P_ch * eta_ch * dt / E_bat;
        P_bat_actual(k) = P_ch;  % Positive values indicate charging
        
    else
        % PV deficit: discharge the battery
        P_dis = min(-P_diff, P_bat_available);
        
        % Update SOC
        SOC = SOC - P_dis / eta_dis * dt / E_bat;
        P_bat_actual(k) = -P_dis;  % Negative values indicate discharging
    end
    
    % Record SOC
    SOC_record(k) = SOC;
    
    % Report progress every 24 hours
    if mod(k, 48) == 0
        fprintf('Completed %.1f days; current SOC: %.1f%%\n', k*dt/24, SOC*100);
    end
end

elapsed_time = toc;
fprintf('\nSimulation completed in %.2f seconds\n', elapsed_time);

%% ================== Analyse results ==================
fprintf('\n========== Simulation results ==========\n');
if cutoff_time >= hours_sim - 0.1
    fprintf('System completed %.1f days of off-grid operation (the full 7 days)\n', hours_sim/24);
    fprintf('  Final SOC: %.1f%%\n', SOC*100);
else
    fprintf('System operated off-grid for %.2f days (%.2f hours)\n', cutoff_time/24, cutoff_time);
    fprintf('  Operating time points: %d / %d\n', cutoff_index, total_points);
    fprintf('  Remaining days: %.2f\n', (hours_sim - cutoff_time)/24);
    fprintf('  SOC at shutdown: %.1f%%\n', SOC*100);
    if energy_deficit > 0
        fprintf('  Energy deficit: %.2f kWh\n', energy_deficit);
    end
end
fprintf('==============================\n');

%% ================== Visual analysis ==================
figure('Position', [50, 50, 1400, 900]);

% Panel 1: power balance
subplot(3,1,1);
plot(t, P_pv_actual, 'g-', 'LineWidth', 1.5); hold on;
plot(t, P_load_actual, 'r-', 'LineWidth', 1.5);
plot(t, P_bat_actual, 'b-', 'LineWidth', 1);
xlabel('Time (hours)');
ylabel('Power (kW)');
title('Power balance');
legend('PV output', 'Load demand', 'Battery power (positive = charge, negative = discharge)', 'Location', 'best');
grid on;
xlim([0, hours_sim]);
ylim([min(P_bat_actual) * 1.1, max([max(P_pv_actual), max(P_load_actual)]) * 1.1]);

% Add a shutdown marker
if cutoff_time < hours_sim
    line([cutoff_time, cutoff_time], get(gca, 'YLim'), 'Color', 'k', 'LineStyle', '--', 'LineWidth', 2);
    text(cutoff_time, max([max(P_pv_actual), max(P_load_actual)])*0.9, ...
         sprintf('Shutdown\n%.1f hours\n(%.2f days)', cutoff_time, cutoff_time/24), ...
         'HorizontalAlignment', 'right', 'FontSize', 10, 'Color', 'red', 'BackgroundColor', 'w');
end

% Panel 2: SOC time series
subplot(3,1,2);
plot(t, SOC_record*100, 'b-', 'LineWidth', 2); hold on;
line([0, hours_sim], [SOC_init*100, SOC_init*100], 'Color', 'g', 'LineStyle', '--', 'LineWidth', 1.5);
line([0, hours_sim], [SOC_min*100, SOC_min*100], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
line([0, hours_sim], [SOC_stop*100, SOC_stop*100], 'Color', 'm', 'LineStyle', '--', 'LineWidth', 1.5);
line([0, hours_sim], [SOC_max*100, SOC_max*100], 'Color', 'c', 'LineStyle', '--', 'LineWidth', 1.5);

% Add a legend
h1 = plot(NaN, NaN, 'b-', 'LineWidth', 2);
h2 = plot(NaN, NaN, '--g', 'LineWidth', 1.5);
h3 = plot(NaN, NaN, '--r', 'LineWidth', 1.5);
h4 = plot(NaN, NaN, '--m', 'LineWidth', 1.5);
h5 = plot(NaN, NaN, '--c', 'LineWidth', 1.5);
legend([h1, h2, h3, h4, h5], ...
       {'SOC', sprintf('Initial SOC (%.0f%%)', SOC_init*100), ...
        sprintf('Lower limit (%.0f%%)', SOC_min*100), ...
        sprintf('Shutdown threshold (%.0f%%)', SOC_stop*100), ...
        sprintf('Upper limit (%.0f%%)', SOC_max*100)}, ...
       'Location', 'best');

xlabel('Time (hours)');
ylabel('SOC (%)');
title('Battery state of charge');
grid on;
xlim([0, hours_sim]);
ylim([0, 100]);

% Add a shutdown marker
if cutoff_time < hours_sim
    line([cutoff_time, cutoff_time], get(gca, 'YLim'), 'Color', 'k', 'LineStyle', '--', 'LineWidth', 2);
end

% Panel 3: cumulative energy
subplot(3,1,3);
% Calculate cumulative energy only up to shutdown
calc_points = min(cutoff_index, total_points);
t_calc = t(1:calc_points);
E_pv_cum = cumtrapz(t_calc, P_pv_actual(1:calc_points)) / 1000;     % MWh
E_load_cum = cumtrapz(t_calc, P_load_actual(1:calc_points)) / 1000;  % MWh
E_bat_cum = cumtrapz(t_calc, abs(P_bat_actual(1:calc_points))) / 1000; % MWh

plot(t_calc, E_pv_cum, 'g-', 'LineWidth', 2); hold on;
plot(t_calc, E_load_cum, 'r-', 'LineWidth', 2);
plot(t_calc, E_bat_cum, 'b-', 'LineWidth', 1.5);

xlabel('Time (hours)');
ylabel('Cumulative energy (MWh)');
title('Cumulative energy');
legend('Total PV generation', 'Total load demand', 'Total battery throughput', 'Location', 'best');
grid on;
xlim([0, hours_sim]);

% Add a shutdown marker
if cutoff_time < hours_sim
    line([cutoff_time, cutoff_time], get(gca, 'YLim'), 'Color', 'k', 'LineStyle', '--', 'LineWidth', 2);
end

% Add an overall title
annotation('textbox', [0 0.95 1 0.05], ...
    'String', sprintf('Off-grid operation | System: peak PV=%.1fkW, battery=%.0fkWh, battery power=%.0fkW', ...
                      P_pv_rated, E_bat, P_bat_max), ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');

%% ================== Detailed data output ==================
fprintf('\n========== Energy statistics ==========\n');
% Include only the period up to shutdown
calc_points = min(cutoff_index, total_points);
E_pv_total = sum(P_pv_actual(1:calc_points)) * dt / 1000;     % MWh
E_load_total = sum(P_load_actual(1:calc_points)) * dt / 1000;  % MWh
E_ch_total = sum(P_bat_actual(P_bat_actual>0)) * dt / 1000;    % MWh
E_dis_total = sum(-P_bat_actual(P_bat_actual<0)) * dt / 1000;   % MWh

fprintf('Operating-period statistics (%.1f hours):\n', cutoff_time);
fprintf('  Total PV generation: %.3f MWh\n', E_pv_total);
fprintf('  Total load demand: %.3f MWh\n', E_load_total);
fprintf('  Total battery charging energy: %.3f MWh\n', E_ch_total);
fprintf('  Total battery discharging energy: %.3f MWh\n', E_dis_total);
if E_load_total > 0
    fprintf('  Energy self-sufficiency: %.1f%%\n', min(100, E_pv_total/E_load_total*100));
end

% Full-period statistics
fprintf('\nFull-period statistics (%.1f hours):\n', hours_sim);
E_pv_total_full = sum(P_pv_raw) * dt / 1000;
E_load_total_full = sum(P_load_raw) * dt / 1000;
fprintf('  Total PV generation: %.3f MWh\n', E_pv_total_full);
fprintf('  Total load demand: %.3f MWh\n', E_load_total_full);
fprintf('  Theoretical self-sufficiency: %.1f%%\n', min(100, E_pv_total_full/E_load_total_full*100));
fprintf('==============================\n');

disp('Analysis completed!');

%% ================== Export data to Excel ==================
% Determine valid data length; save only up to shutdown if it occurs early
if cutoff_index < total_points
    valid_len = cutoff_index;
else
    valid_len = total_points;
end

% Extract data for the valid time period
t_valid = t(1:valid_len)';                      % Time (hours)
P_pv_valid = P_pv_actual(1:valid_len);          % PV power (kW)
P_load_valid = P_load_actual(1:valid_len);      % Load power (kW)
P_bat_valid = P_bat_actual(1:valid_len);        % Battery power (kW), positive for charging and negative for discharging
SOC_bat_valid = SOC_record(1:valid_len) * 100;  % Battery SOC (%)

% Define table variable names in English
col_names = {'Time_h', 'PV_kW', 'Load_kW', 'Battery_kW', ...
            'Battery_SOC_pct'};

% Combine the data into a table
data_table = table(t_valid, P_pv_valid, P_load_valid, P_bat_valid, ...
                   SOC_bat_valid, 'VariableNames', col_names);

% Set the output filename; adjust the path as needed
filename = 'simulation_results.xlsx';

% Write the Excel file
try
    writetable(data_table, filename, 'Sheet', 'SimulationResults');
    fprintf('\nData saved successfully to: %s\n', filename);
    fprintf('  Variables: time (hours), PV (kW), load (kW), battery power (kW), battery SOC (%%)\n');
catch ME
    warning('Failed to save Excel file: %s', ME.message);
    % Fallback: save a CSV file
    csv_filename = 'simulation_results.csv';
    writetable(data_table, csv_filename);
    fprintf('Saved as CSV instead: %s\n', csv_filename);
end

disp('Data export completed!');