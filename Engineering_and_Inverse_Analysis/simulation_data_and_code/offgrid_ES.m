clc;
clear;
close all;

%% ================== 系统参数设置 ==================
% 电池参数
SOC_init = 0.713;      % 初始荷电状态 70%
SOC_min = 0.1;       % 正常运行下限 10%
SOC_stop = 0.1;      % 停止阈值 10%
SOC_max = 0.9;       % 最大充电限制 90%

% 效率参数
eta_ch = 0.95;         % 充电效率
eta_dis = 0.95;        % 放电效率

% 时间参数
dt = 0.5;             % 时间步长 0.5小时（30分钟）
total_points = 336;   % 总数据点数（7天×48点/天）
hours_sim = total_points * dt;  % 模拟时长 168小时 (7天)

%% ================== 读取数据 ==================
% 读取7天数据
data = xlsread('data_7day.xlsx');

% 提取数据（第1列:光伏功率, 第2列:负荷功率）
P_pv_raw = data(:,1);      % 原始光伏功率 (kW)
P_load_raw = data(:,2);    % 原始负荷功率 (kW)

% 检查数据完整性
fprintf('========== 数据信息 ==========\n');
fprintf('原始数据点数量: %d\n', length(P_pv_raw));
fprintf('数据时间步长: %.1f 小时\n', dt);
fprintf('数据时间范围: %.1f 小时 (%.1f 天)\n', length(P_pv_raw)*dt, length(P_pv_raw)*dt/24);
fprintf('光伏功率范围: %.2f - %.2f kW\n', min(P_pv_raw), max(P_pv_raw));
fprintf('负荷功率范围: %.2f - %.2f kW\n', min(P_load_raw), max(P_load_raw));
fprintf('===============================\n\n');

% 确保数据足够168小时
if length(P_pv_raw) < total_points
    warning('数据不足！实际只有 %.0f 个点，需要 %d 个点', length(P_pv_raw), total_points);
    total_points = length(P_pv_raw);
    hours_sim = total_points * dt;
end

% 只取需要的数据点
P_pv_raw = P_pv_raw(1:total_points);
P_load_raw = P_load_raw(1:total_points);

% 创建时间序列（半小时级）
t = (0:total_points-1) * dt;  % 时间点（小时）

fprintf('使用数据点: %d 个\n', total_points);
fprintf('模拟时长: %.1f 小时 (%.1f 天)\n', hours_sim, hours_sim/24);

%% ================== 系统容量配置 ==================
% 根据实际数据配置系统容量
P_pv_rated = max(P_pv_raw);     % 光伏额定功率 (kW)
% P_bat_max = 3000;                % 电池最大功率 (kW) - 用户设定
% E_bat = 6000;                    % 电池容量 (kWh) - 用户设定
P_bat_max = 5800;                % 电池最大功率 (kW) - 用户设定
E_bat = 11600;                    % 电池容量 (kWh) - 用户设定

fprintf('\n========== 系统配置 ==========\n');
fprintf('光伏峰值功率: %.2f kW\n', max(P_pv_raw));
fprintf('光伏平均功率: %.2f kW\n', mean(P_pv_raw));
fprintf('光伏额定功率: %.2f kW\n', P_pv_rated);
fprintf('电池额定容量: %.2f kWh\n', E_bat);
fprintf('电池最大功率: %.2f kW\n', P_bat_max);
fprintf('初始SOC: %.1f%%\n', SOC_init*100);
fprintf('==============================\n\n');

%% ================== 主仿真循环 ==================
% 初始化
SOC = SOC_init;                  % 初始SOC
P_pv_actual = zeros(total_points, 1);     % 实际光伏出力
P_load_actual = zeros(total_points, 1);   % 实际负荷
P_bat_actual = zeros(total_points, 1);    % 实际电池功率（正=充电，负=放电）
SOC_record = zeros(total_points, 1);      % SOC记录
SOC_record(1) = SOC;

% 状态变量
is_running = true;               % 系统运行状态
cutoff_time = hours_sim;         % 断电时间（初始为满时长）
cutoff_index = total_points;     % 断电时刻索引
energy_deficit = 0;              % 能量缺额累计

% 仿真开始
fprintf('开始仿真 (总时长: %.1f 小时, %d 个时间点)...\n', hours_sim, total_points);
tic;

for k = 1:total_points
    if ~is_running
        break;  % 系统已断电，停止仿真
    end
    
    % 当前时刻的功率（直接使用原始数据，不需要缩放）
    P_pv = P_pv_raw(k);    % 实际光伏功率 (kW)
    P_load = P_load_raw(k); % 实际负荷功率 (kW)
    
    % 存储实际值
    P_pv_actual(k) = P_pv;
    P_load_actual(k) = P_load;
    
    %% ===== 计算可用放电能力 =====
    if SOC > SOC_min
        % 正常运行区：可用容量为当前SOC到下限
        E_dis_max = (SOC - SOC_min) * E_bat * eta_dis;
    else
        % 应急透支区：可用容量为当前SOC到停止阈值
        E_dis_max = (SOC - SOC_stop) * E_bat * eta_dis;
    end
    
    % 最大放电功率（受电池功率限制和能量限制）
    P_bat_available = min(P_bat_max, E_dis_max / dt);
    
    %% ===== 停止条件检查 =====
    if (P_pv + P_bat_available < P_load) || (SOC <= SOC_stop)
        is_running = false;
        cutoff_time = t(k);
        cutoff_index = k;
        energy_deficit = (P_load - P_pv - P_bat_available) * dt;
        fprintf('\n系统在第 %.2f 小时断电 (%.2f 天)\n', cutoff_time, cutoff_time/24);
        fprintf('断电时刻: 时间点=%d, 光伏=%.2f kW, 负荷=%.2f kW, SOC=%.2f%%\n', ...
                k, P_pv, P_load, SOC*100);
        break;
    end
    
    %% ===== 功率平衡与电池充放电策略 =====
    P_diff = P_pv - P_load;  % 功率差额
    
    if P_diff >= 0
        % 光伏过剩：充电
        % 限制充电功率不超过电池最大功率
        P_ch = min(P_diff, P_bat_max);
        
        % 限制充电功率不超过可用容量
        E_ch_max = (SOC_max - SOC) * E_bat / eta_ch;
        P_ch = min(P_ch, E_ch_max / dt);
        
        % 更新SOC
        SOC = SOC + P_ch * eta_ch * dt / E_bat;
        P_bat_actual(k) = P_ch;  % 正值为充电
        
    else
        % 光伏不足：放电
        P_dis = min(-P_diff, P_bat_available);
        
        % 更新SOC
        SOC = SOC - P_dis / eta_dis * dt / E_bat;
        P_bat_actual(k) = -P_dis;  % 负值为放电
    end
    
    % 记录SOC
    SOC_record(k) = SOC;
    
    % 进度显示（每24小时显示一次）
    if mod(k, 48) == 0
        fprintf('已完成 %.1f 天, 当前SOC: %.1f%%\n', k*dt/24, SOC*100);
    end
end

elapsed_time = toc;
fprintf('\n仿真完成，耗时: %.2f 秒\n', elapsed_time);

%% ================== 结果分析 ==================
fprintf('\n========== 仿真结果 ==========\n');
if cutoff_time >= hours_sim - 0.1
    fprintf('? 系统成功离网运行 %.1f 天 (满7天)\n', hours_sim/24);
    fprintf('  最终SOC: %.1f%%\n', SOC*100);
else
    fprintf('? 系统离网运行了 %.2f 天 (%.2f 小时)\n', cutoff_time/24, cutoff_time);
    fprintf('  运行时间点: %d / %d\n', cutoff_index, total_points);
    fprintf('  剩余天数: %.2f 天\n', (hours_sim - cutoff_time)/24);
    fprintf('  断电时SOC: %.1f%%\n', SOC*100);
    if energy_deficit > 0
        fprintf('  缺额能量: %.2f kWh\n', energy_deficit);
    end
end
fprintf('==============================\n');

%% ================== 可视化分析 ==================
figure('Position', [50, 50, 1400, 900]);

% 子图1：功率平衡图
subplot(3,1,1);
plot(t, P_pv_actual, 'g-', 'LineWidth', 1.5); hold on;
plot(t, P_load_actual, 'r-', 'LineWidth', 1.5);
plot(t, P_bat_actual, 'b-', 'LineWidth', 1);
xlabel('时间 (小时)');
ylabel('功率 (kW)');
title('功率平衡分析');
legend('光伏出力', '负荷需求', '电池功率 (正=充,负=放)', 'Location', 'best');
grid on;
xlim([0, hours_sim]);
ylim([min(P_bat_actual) * 1.1, max([max(P_pv_actual), max(P_load_actual)]) * 1.1]);

% 添加断电标记
if cutoff_time < hours_sim
    line([cutoff_time, cutoff_time], get(gca, 'YLim'), 'Color', 'k', 'LineStyle', '--', 'LineWidth', 2);
    text(cutoff_time, max([max(P_pv_actual), max(P_load_actual)])*0.9, ...
         sprintf('断电时刻\n%.1f小时\n(%.2f天)', cutoff_time, cutoff_time/24), ...
         'HorizontalAlignment', 'right', 'FontSize', 10, 'Color', 'red', 'BackgroundColor', 'w');
end

% 子图2：SOC变化曲线
subplot(3,1,2);
plot(t, SOC_record*100, 'b-', 'LineWidth', 2); hold on;
line([0, hours_sim], [SOC_init*100, SOC_init*100], 'Color', 'g', 'LineStyle', '--', 'LineWidth', 1.5);
line([0, hours_sim], [SOC_min*100, SOC_min*100], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
line([0, hours_sim], [SOC_stop*100, SOC_stop*100], 'Color', 'm', 'LineStyle', '--', 'LineWidth', 1.5);
line([0, hours_sim], [SOC_max*100, SOC_max*100], 'Color', 'c', 'LineStyle', '--', 'LineWidth', 1.5);

% 添加图例
h1 = plot(NaN, NaN, 'b-', 'LineWidth', 2);
h2 = plot(NaN, NaN, '--g', 'LineWidth', 1.5);
h3 = plot(NaN, NaN, '--r', 'LineWidth', 1.5);
h4 = plot(NaN, NaN, '--m', 'LineWidth', 1.5);
h5 = plot(NaN, NaN, '--c', 'LineWidth', 1.5);
legend([h1, h2, h3, h4, h5], ...
       {'SOC', sprintf('初始SOC(%.0f%%)', SOC_init*100), ...
        sprintf('下限(%.0f%%)', SOC_min*100), ...
        sprintf('停止阈值(%.0f%%)', SOC_stop*100), ...
        sprintf('上限(%.0f%%)', SOC_max*100)}, ...
       'Location', 'best');

xlabel('时间 (小时)');
ylabel('SOC (%)');
title('电池荷电状态变化');
grid on;
xlim([0, hours_sim]);
ylim([0, 100]);

% 添加断电标记
if cutoff_time < hours_sim
    line([cutoff_time, cutoff_time], get(gca, 'YLim'), 'Color', 'k', 'LineStyle', '--', 'LineWidth', 2);
end

% 子图3：累计能量分析
subplot(3,1,3);
% 只计算到断电时刻的累计能量
calc_points = min(cutoff_index, total_points);
t_calc = t(1:calc_points);
E_pv_cum = cumtrapz(t_calc, P_pv_actual(1:calc_points)) / 1000;     % MWh
E_load_cum = cumtrapz(t_calc, P_load_actual(1:calc_points)) / 1000;  % MWh
E_bat_cum = cumtrapz(t_calc, abs(P_bat_actual(1:calc_points))) / 1000; % MWh

plot(t_calc, E_pv_cum, 'g-', 'LineWidth', 2); hold on;
plot(t_calc, E_load_cum, 'r-', 'LineWidth', 2);
plot(t_calc, E_bat_cum, 'b-', 'LineWidth', 1.5);

xlabel('时间 (小时)');
ylabel('累计能量 (MWh)');
title('累计能量分析');
legend('光伏总发电量', '负荷总需求量', '电池总吞吐量', 'Location', 'best');
grid on;
xlim([0, hours_sim]);

% 添加断电标记
if cutoff_time < hours_sim
    line([cutoff_time, cutoff_time], get(gca, 'YLim'), 'Color', 'k', 'LineStyle', '--', 'LineWidth', 2);
end

% 添加总标题
annotation('textbox', [0 0.95 1 0.05], ...
    'String', sprintf('离网运行分析 | 系统配置: PV峰值=%.1fkW, 电池=%.0fkWh, 电池功率=%.0fkW', ...
                      P_pv_rated, E_bat, P_bat_max), ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');

%% ================== 详细数据输出 ==================
fprintf('\n========== 能量统计 ==========\n');
% 只统计到断电时刻
calc_points = min(cutoff_index, total_points);
E_pv_total = sum(P_pv_actual(1:calc_points)) * dt / 1000;     % MWh
E_load_total = sum(P_load_actual(1:calc_points)) * dt / 1000;  % MWh
E_ch_total = sum(P_bat_actual(P_bat_actual>0)) * dt / 1000;    % MWh
E_dis_total = sum(-P_bat_actual(P_bat_actual<0)) * dt / 1000;   % MWh

fprintf('运行时段统计 (%.1f 小时):\n', cutoff_time);
fprintf('  光伏总发电量: %.3f MWh\n', E_pv_total);
fprintf('  负荷总需求量: %.3f MWh\n', E_load_total);
fprintf('  电池总充电量: %.3f MWh\n', E_ch_total);
fprintf('  电池总放电量: %.3f MWh\n', E_dis_total);
if E_load_total > 0
    fprintf('  能量自给率: %.1f%%\n', min(100, E_pv_total/E_load_total*100));
end

% 全时段统计
fprintf('\n全时段统计 (%.1f 小时):\n', hours_sim);
E_pv_total_full = sum(P_pv_raw) * dt / 1000;
E_load_total_full = sum(P_load_raw) * dt / 1000;
fprintf('  光伏总发电量: %.3f MWh\n', E_pv_total_full);
fprintf('  负荷总需求量: %.3f MWh\n', E_load_total_full);
fprintf('  理论自给率: %.1f%%\n', min(100, E_pv_total_full/E_load_total_full*100));
fprintf('==============================\n');

disp('分析完成！');

%% ================== 导出数据到Excel ==================
% 确定有效数据长度（若提前断电，只保存到断电时刻）
if cutoff_index < total_points
    valid_len = cutoff_index;
else
    valid_len = total_points;
end

% 提取有效时间段的数据
t_valid = t(1:valid_len)';                      % 时间 (小时)
P_pv_valid = P_pv_actual(1:valid_len);          % 光伏功率 (kW)
P_load_valid = P_load_actual(1:valid_len);      % 负荷功率 (kW)
P_bat_valid = P_bat_actual(1:valid_len);        % 电池功率 (kW) 正值充电，负值放电
SOC_bat_valid = SOC_record(1:valid_len) * 100;  % 电池SOC (%)

% 创建表格变量名（全部使用英文）
col_names = {'Time_h', 'PV_kW', 'Load_kW', 'Battery_kW', ...
            'Battery_SOC_pct'};

% 将数据组合成表格
data_table = table(t_valid, P_pv_valid, P_load_valid, P_bat_valid, ...
                   SOC_bat_valid, 'VariableNames', col_names);

% 指定保存的文件名（可根据需要修改路径）
filename = 'simulation_results.xlsx';

% 写入Excel文件
try
    writetable(data_table, filename, 'Sheet', 'SimulationResults');
    fprintf('\n数据已成功保存到文件: %s\n', filename);
    fprintf('  包含变量: 时间(小时), 光伏(kW), 负荷(kW), 电池功率(kW), 电池SOC(%%)\n');
catch ME
    warning('保存Excel文件失败: %s', ME.message);
    % 备选方案：保存为CSV文件
    csv_filename = 'simulation_results.csv';
    writetable(data_table, csv_filename);
    fprintf('已改为保存为CSV文件: %s\n', csv_filename);
end

disp('数据导出完成！');