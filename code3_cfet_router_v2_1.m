% main_script.m
% This script will orchestrate the entire process.
% main_script.m
% 该脚本将协调整个过程。
function code3_cfet_router_v2_1(input_data_cell,design_rule)
% 定义函数 code3_cfet_router_v2_1，它接受两个输入参数：input_data_cell 和 design_rule。

% --- Parameters ---
% --- 参数 ---
track_height = 5; % Default or user-settable
% 设置布线层的高度（track_height），默认为 5。这是一个可调参数。

% --- Input Data (Example from prompt) ---
% --- 输入数据（来自提示的示例） ---
% Ensure input_data is a numeric matrix. Handle NaNs appropriately for kruskal.
% 确保 input_data 是一个数值矩阵。为 Kruskal 算法适当地处理 NaN。
% input_data_cell = {
%     [2, 1, 5, 6, 2, 9, 8, 5, 2, 8, 10];
%     [2, 1, 3, NaN, 2, 3, 7, 6, 2, 7, 10];
%     [5, 1, 11, 6, 4, 9, 13, 5, 8, NaN, 14, 8, 4];
%     [7, 3, 12, 6, 4, 1, 3, NaN, 10, 7, 14]
% };
% 这是 input_data_cell 的示例结构，它是一个包含多个行向量的单元数组。

% Determine max L and pad shorter rows with NaN for consistent L
% 确定最大 L（列数），并用 NaN 填充较短的行以保持一致的 L。
max_L_initial = 0;
% 初始化最大 L 为 0。
for i = 1:numel(input_data_cell)
% 遍历 input_data_cell 中的每个单元格（行）。
    if length(input_data_cell{i}) > max_L_initial
    % 如果当前行的长度大于 max_L_initial。
        max_L_initial = length(input_data_cell{i});
        % 更新 max_L_initial 为当前行的长度。
    end
end
input_data_matrix = nan(4, max_L_initial);
% 创建一个 4xmax_L_initial 的矩阵 input_data_matrix，并用 NaN 填充。
for i = 1:4
% 遍历 input_data_cell 的前 4 行。
    row_len = length(input_data_cell{i});
    % 获取当前行的长度。
    input_data_matrix(i, 1:row_len) = input_data_cell{i};
    % 将 input_data_cell 的数据复制到 input_data_matrix 中，对齐到左侧。
end
L = max_L_initial;
% 将 L 设置为确定的最大列数。

% --- 1. Initialize Layout ---
% --- 1. 初始化布局 ---
fprintf('1. Initializing Layout...\n');
% 打印初始化布局的提示信息。
layer_info = define_layer_info(track_height, L);
% 调用 define_layer_info 函数，根据 track_height 和 L 定义层信息。
layout = initialize_layout(input_data_matrix, track_height, L, layer_info);
% 调用 initialize_layout 函数，根据输入数据和层信息初始化布局。
fprintf('Layout Initialized.\n\n');
% 打印布局初始化完成的提示信息。

% --- Design Rule for Kruskal MST (example values) ---
% --- Kruskal MST 的设计规则（示例值） ---
% design_rule = struct('a',1, 'b',2, 'c',3, 'd',100, ...
%                      'e1',1,'e2',1, 'f1',2,'f2',1, ...
%                      'g1',3,'g2',1, 'h1',4,'h2',1, 'i',0.1);
% 这是一个 design_rule 结构体的示例，包含了不同布线规则的成本参数。

% --- 2. Identify Nets and Target Connections using User's Kruskal functions ---
% --- 2. 使用用户提供的 Kruskal 函数识别网络和目标连接 ---
fprintf('2. Identifying Nets and Target Connections...\n');
% 打印识别网络和目标连接的提示信息。
% The 'code2_calculate_total_weight' function internally uses 'kruskal_mst'.
% 'code2_calculate_total_weight' 函数内部使用 'kruskal_mst'。
% We need to modify/wrap 'kruskal_mst' to return the actual MST edges.
% 我们需要修改/包装 'kruskal_mst' 以返回实际的 MST 边缘。
% Modified kruskal_mst to return edges
% 修改后的 kruskal_mst 以返回边缘。
[nets_to_route, all_initial_pins] = get_routing_tasks_from_input(input_data_matrix, design_rule, track_height, layer_info);
% 调用 get_routing_tasks_from_input 函数，根据输入数据、设计规则和层信息，获取需要布线的网络和所有初始引脚信息。
fprintf('%d unique potentials with multiple pins identified.\n', numel(nets_to_route));
% 打印识别到的具有多个引脚的唯一电位的数量。
for i=1:numel(nets_to_route)
% 遍历每个需要布线的网络。
    fprintf('  Potential %d needs %d segments to be routed.\n', nets_to_route(i).potential, size(nets_to_route(i).segments,1));
    % 打印每个电位需要布线的段的数量。
end
fprintf('Nets Identified.\n\n');
% 打印网络识别完成的提示信息。

% --- 3. Iterative Routing ---
% --- 3. 迭代布线 ---
fprintf('3. Starting Iterative Routing...\n');
% 打印开始迭代布线的提示信息。
max_routing_iterations = 5; % Main loop for rip-up and reroute
% 设置最大布线迭代次数为 5。这是用于拆除和重新布线的主循环。
max_attempts_per_segment = 3; % Attempts for a single segment before trying rip-up
% 设置每个线段在尝试拆除和重新布线之前的最大尝试次数为 3。

% Add status and attempt counts to nets_to_route segments
% 为 nets_to_route 中的线段添加状态和尝试计数。
for i = 1:numel(nets_to_route)
% 遍历每个需要布线的网络。
    num_segments = size(nets_to_route(i).segments, 1);
    % 获取当前网络的线段数量。
    nets_to_route(i).segment_status = cell(num_segments, 1); % 'pending', 'routed', 'failed'
    % 初始化线段状态，可以是 'pending'（待处理）、'routed'（已布线）或 'failed'（失败）。
    nets_to_route(i).segment_paths = cell(num_segments, 1);
    % 初始化线段路径。
    nets_to_route(i).segment_attempts = zeros(num_segments, 1);
    % 初始化线段尝试次数为 0。
    for j=1:num_segments
    % 遍历当前网络的每个线段。
        nets_to_route(i).segment_status{j} = 'pending';
        % 将当前线段的状态设置为 'pending'。
    end
end

for iter = 1:max_routing_iterations
% 开始迭代布线主循环。
    fprintf('--- Routing Iteration %d ---\n', iter);
    % 打印当前迭代次数。
    segments_routed_this_iteration = 0;
    % 初始化本轮迭代中成功布线的线段数量。
    has_pending_segments = false;
    % 标记是否有待处理的线段。

    for net_idx = 1:numel(nets_to_route)
    % 遍历每个需要布线的网络。
        potential = nets_to_route(net_idx).potential;
        % 获取当前网络的电位值。
        for seg_idx = 1:size(nets_to_route(net_idx).segments, 1)
        % 遍历当前网络的每个线段。
            if strcmp(nets_to_route(net_idx).segment_status{seg_idx}, 'routed')
            % 如果当前线段已经布线。
                continue; % Already routed
                % 跳过，处理下一个线段。
            end
            has_pending_segments = true;
            % 标记存在待处理的线段。
            if nets_to_route(net_idx).segment_attempts(seg_idx) >= max_attempts_per_segment
            % 如果当前线段的尝试次数达到上限。
                continue; % Tried too many times for this segment in its current state
                % 跳过，处理下一个线段。
            end
            nets_to_route(net_idx).segment_attempts(seg_idx) = nets_to_route(net_idx).segment_attempts(seg_idx) + 1;
            % 增加当前线段的尝试次数。
            segment = nets_to_route(net_idx).segments(seg_idx,:); % [p1_idx, p2_idx] referring to all_initial_pins
            % 获取当前线段的两个引脚索引。
            pin1_coord_3d = all_initial_pins(segment(1)).coord_3d;
            % 获取第一个引脚的 3D 坐标。
            pin2_coord_3d = all_initial_pins(segment(2)).coord_3d;
            % 获取第二个引脚的 3D 坐标。
            fprintf('Attempting to route Potential %d: (%d,%d,%s) to (%d,%d,%s)\n', potential, ...
                pin1_coord_3d(1), pin1_coord_3d(2), layer_info.names{pin1_coord_3d(3)}, ...
                pin2_coord_3d(1), pin2_coord_3d(2), layer_info.names{pin2_coord_3d(3)});
            % 打印当前尝试布线的线段信息。

            % --- A* Pathfinding ---
            % --- A* 路径查找 ---
            % The A* should find path from any point of current potential connected to pin1_coord_3d
            % to pin2_coord_3d, or vice-versa.
            % A* 应该从连接到 pin1_coord_3d 的当前电位的任何点找到到 pin2_coord_3d 的路径，反之亦然。
            % For simplicity here, we assume points are distinct and try to connect them directly.
            % 为简单起见，这里假设点是不同的并尝试直接连接它们。
            % A more robust A* takes the entire existing net component as a possible start.
            % 更健壮的 A* 会将整个现有网络组件作为可能的起点。
            [path_coords, ~] = pathfinder_astar(layout, pin1_coord_3d, pin2_coord_3d, potential, layer_info, L, track_height);
            % 调用 pathfinder_astar 函数查找从 pin1 到 pin2 的路径。
            if ~isempty(path_coords)
            % 如果找到了路径。
                % Apply path
                % 应用路径
                [layout, success_apply] = apply_path_to_layout(layout, path_coords, potential, layer_info);
                % 调用 apply_path_to_layout 函数将路径应用到布局上。
                if success_apply
                % 如果路径应用成功。
                    nets_to_route(net_idx).segment_status{seg_idx} = 'routed';
                    % 将线段状态设置为 'routed'。
                    nets_to_route(net_idx).segment_paths{seg_idx} = path_coords;
                    % 保存线段的路径。
                    segments_routed_this_iteration = segments_routed_this_iteration + 1;
                    % 增加本轮成功布线的线段数量。
                    fprintf('  SUCCESSFULLY Routed.\n');
                    % 打印布线成功信息。
                else
                    nets_to_route(net_idx).segment_status{seg_idx} = 'failed_apply_conflict';
                    % 设置线段状态为 'failed_apply_conflict'（应用冲突失败）。
                     fprintf('  FAILED to apply path (conflict during apply).\n');
                     % 打印应用路径失败（冲突）信息。
                end
            else
                nets_to_route(net_idx).segment_status{seg_idx} = 'failed_no_path';
                % 设置线段状态为 'failed_no_path'（未找到路径失败）。
                fprintf('  FAILED to find path.\n');
                % 打印未找到路径信息。
                % --- Rip-up and Reroute (Basic Strategy) ---
                % --- 拆除和重新布线（基本策略） ---
                if iter < max_routing_iterations % Don't rip-up on the last iteration usually
                % 如果不是最后一轮迭代（通常不在最后一轮拆除）。
                    % Identify a blocking net (this is complex; A* should ideally give hints)
                    % 识别阻塞网络（这很复杂；A* 理想情况下应提供提示）。
                    % For simplicity: if path for P_A fails, and it crosses P_B, try ripping P_B.
                    % 为简单起见：如果 P_A 的路径失败，并且它穿过 P_B，尝试拆除 P_B。
                    % This requires detailed conflict analysis from A* or a probing mechanism.
                    % 这需要 A* 的详细冲突分析或探测机制。
                    % As a placeholder: if a segment fails, subsequent iterations might resolve it if other segments move.
                    % 作为占位符：如果一个线段失败，如果其他线段移动，后续迭代可能会解决它。
                    % A more active rip-up would be:
                    % 更主动的拆除策略将是：
                    % [blocker_net_idx, blocker_seg_idx] = find_blocker_for_segment(layout, pin1_coord_3d, pin2_coord_3d, layer_info, nets_to_route);
                    % if ~isempty(blocker_net_idx)
                    %    fprintf('   Attempting to rip-up blocking segment from potential %d\n', nets_to_route(blocker_net_idx).potential);
                    %    layout = remove_path_from_layout(layout, nets_to_route(blocker_net_idx).segment_paths{blocker_seg_idx}, layer_info);
                    %    nets_to_route(blocker_net_idx).segment_status{blocker_seg_idx} = 'pending_ripped';
                    %    nets_to_route(blocker_net_idx).segment_attempts(blocker_seg_idx) = 0; % Reset attempts for ripped segment
                    % end
                    % 以上是被注释掉的更复杂的拆除和重新布线逻辑。
                end
            end
        end
    end
    if ~has_pending_segments
    % 如果没有待处理的线段。
         fprintf('All segments appear routed or maxed out attempts after iteration %d.\n', iter);
         % 打印所有线段已布线或尝试次数已满的信息。
         break;
         % 退出迭代循环。
    end
    if segments_routed_this_iteration == 0 && iter > 1 % Check if any progress was made
    % 如果本轮迭代没有成功布线任何线段，且不是第一次迭代。
        fprintf('No segments were successfully routed in iteration %d. May indicate persistent blockage.\n', iter);
        % 打印没有线段成功布线的信息，可能表示持续阻塞。
        % Could stop early, or continue if rip-up is more aggressive.
        % 可以提前停止，或者如果拆除策略更激进则继续。
    end
end
fprintf('Routing Phase Completed.\n\n');
% 打印布线阶段完成的提示信息。

% --- 4. Final Connectivity Check ---
% --- 4. 最终连接性检查 ---
fprintf('4. Performing Final Connectivity Checks...\n');
% 打印执行最终连接性检查的提示信息。
all_potentials_in_layout = [];
% 初始化布局中所有电位的列表。
for i = 1:numel(all_initial_pins) % Get all relevant potentials
% 遍历所有初始引脚，获取所有相关电位。
    all_potentials_in_layout = [all_potentials_in_layout, all_initial_pins(i).potential];
    % 将当前引脚的电位添加到列表中。
end
unique_potentials = unique(all_potentials_in_layout);
% 获取唯一的电位值。
for i = 1:numel(unique_potentials)
% 遍历每个唯一的电位。
    p_val = unique_potentials(i);
    % 获取当前电位值。
    if isnan(p_val), continue; end
    % 如果电位是 NaN，则跳过。
    num_initial_pins_for_potential = 0;
    % 初始化当前电位的初始引脚数量。
    for k=1:numel(all_initial_pins)
    % 遍历所有初始引脚。
        if all_initial_pins(k).potential == p_val
        % 如果当前引脚的电位与当前检查的电位相同。
            num_initial_pins_for_potential = num_initial_pins_for_potential + 1;
            % 增加初始引脚数量。
        end
    end
    if num_initial_pins_for_potential <= 1
    % 如果当前电位的初始引脚数量小于等于 1。
        fprintf('Potential %d: has %d pin(s), no complex connectivity check needed.\n', p_val, num_initial_pins_for_potential);
        % 打印无需复杂连接性检查的信息。
        continue;
        % 跳过，处理下一个电位。
    end
    [is_connected, component_size, total_size] = check_potential_connectivity_bfs(layout, p_val, layer_info);
    % 调用 check_potential_connectivity_bfs 函数检查当前电位的连接性。
    if is_connected
    % 如果电位完全连接。
        fprintf('Potential %d: IS CONNECTED. (Component size: %d, Total cells: %d)\n', p_val, component_size, total_size);
        % 打印电位已连接的信息。
    else
        fprintf('Potential %d: IS NOT FULLY CONNECTED. (Component size: %d, Total cells: %d)\n', p_val, component_size, total_size);
        % 打印电位未完全连接的信息。
        % Here, one could trigger more advanced global or local rerouting diagnostics/attempts.
        % 在这里，可以触发更高级的全局或局部重新布线诊断/尝试。
    end
end
fprintf('Connectivity Checks Completed.\n\n');
% 打印连接性检查完成的提示信息。

% --- 5. Visualization ---
% --- 5. 可视化 ---
fprintf('5. Visualizing Layout...\n');
% 打印可视化布局的提示信息。
visualize_layout(layout, layer_info, unique_potentials);
% 调用 visualize_layout 函数可视化布局。
fprintf('Visualization Generated.\n');
% 打印可视化生成完成的提示信息。
end
% 函数结束。

% --- Helper function: Define Layer Information ---
% --- 辅助函数：定义层信息 ---
function layer_info = define_layer_info(track_height, L)
% 定义函数 define_layer_info，它接受 track_height 和 L 作为输入。
    layer_names = {
        'FM2', 'FV_M1_M2', 'FM1', 'FV_M0_M1', 'FM0', 'FV_MD_M0', 'FMD', ...
        'V_FMD_BMD', ...
        'BMD', 'BV_MD_M0', 'BM0', 'BV_M0_M1', 'BM1', 'BV_M1_M2', 'BM2'
    };
    % 定义所有层的名称，包括金属层（F/B）和过孔层（FV/BV/V）。
    % H: Horizontal metal, V: Vertical metal, VIA: Via
    % H: 水平金属层, V: 垂直金属层, VIA: 过孔层
    layer_types = {
        'H', 'VIA', 'V', 'VIA', 'H', 'VIA', 'V', ... % Top stack (F)
        'VIA', ...                                  % Connecting FMD & BMD
        'V', 'VIA', 'H', 'VIA', 'V', 'VIA', 'H'     % Bottom stack (B)
    };
    % 定义对应层的类型（水平、垂直或过孔）。
    layer_info.names = layer_names;
    % 将层名称存储到 layer_info 结构体中。
    layer_info.types = layer_types;
    % 将层类型存储到 layer_info 结构体中。
    layer_info.track_height = track_height;
    % 将 track_height 存储到 layer_info 结构体中。
    layer_info.L = L;
    % 将 L 存储到 layer_info 结构体中。
    layer_info.name_to_idx = containers.Map(layer_names, 1:numel(layer_names));
    % 创建一个从层名称到索引的映射。
    layer_info.idx_to_name = containers.Map(1:numel(layer_names), layer_names);
    % 创建一个从层索引到名称的映射。
end

% --- Helper function: Initialize Layout ---
% --- 辅助函数：初始化布局 ---
function layout = initialize_layout(input_data, track_height, L, layer_info)
% 定义函数 initialize_layout，它接受 input_data、track_height、L 和 layer_info 作为输入。
    layout = struct();
    % 初始化 layout 结构体。
    for i = 1:numel(layer_info.names)
    % 遍历所有层。
        layout.(layer_info.names{i}) = ones(track_height, L) * -1; % -1 for unused
        % 为每层创建一个 track_height x L 的矩阵，并用 -1（表示未使用）填充。
    end
    % Fill initial data
    % 填充初始数据
    % input_data(1,:) -> layout.FMD (row 2)
    % input_data(2,:) -> layout.BMD (row 2)
    % input_data(3,:) -> layout.FMD (row track_height-1)
    % input_data(4,:) -> layout.BMD (row track_height-1)
    if size(input_data,2) == L
    % 如果 input_data 的列数与 L 相等。
        layout.FMD(2, :) = input_data(1, :);
        % 将 input_data 的第一行放置到 FMD 层的第 2 行。
        layout.BMD(2, :) = input_data(2, :);
        % 将 input_data 的第二行放置到 BMD 层的第 2 行。
        layout.FMD(track_height - 1, :) = input_data(3, :);
        % 将 input_data 的第三行放置到 FMD 层的倒数第二行。
        layout.BMD(track_height - 1, :) = input_data(4, :);
        % 将 input_data 的第四行放置到 BMD 层的倒数第二行。
    else
        warning('Input data L does not match layout L. Skipping initial data placement in FMD/BMD rows from input_data.');
        % 如果 input_data 的列数不匹配布局的 L，则发出警告。
    end
    % layout.V_FMD_BMD (rows 2 to track_height-1) are -2 (unusable for routing through these cells initially)
    % layout.V_FMD_BMD (第 2 行到 track_height-1 行) 被设置为 -2（初始时这些单元格不能用于布线）。
    % However, the problem statement says "V_FMD_BMD连接FMD和BMD".
    % 然而，问题描述说 "V_FMD_BMD 连接 FMD 和 BMD"。
    % This implies these cells *are* the vias. They are not "unusable space" but rather the via layer itself.
    % 这意味着这些单元格 *就是* 过孔。它们不是“不可用空间”，而是过孔层本身。
    % The rule is: if FMD(r,c) and BMD(r,c) need to connect, then V_FMD_BMD(r,c) must also be part of that net.
    % 规则是：如果 FMD(r,c) 和 BMD(r,c) 需要连接，那么 V_FMD_BMD(r,c) 也必须是该网络的一部分。
    % The constraint is on *which rows* of V_FMD_BMD can be used.
    % 约束在于 V_FMD_BMD 的 *哪些行* 可以使用。
    % "layout.V_FMD_BMD的第二行到倒数第二行填-2" - this means these specific via locations CANNOT be used.
    % "layout.V_FMD_BMD 的第二行到倒数第二行填 -2" - 这意味着这些特定的过孔位置不能使用。
    % This is a strong constraint. Let's follow it.
    % 这是一个强约束。我们遵循它。
    if track_height >= 3
    % 如果 track_height 大于等于 3。
         layout.V_FMD_BMD(2:(track_height-1), :) = -2;
         % 将 V_FMD_BMD 层的第 2 行到倒数第二行设置为 -2（表示不可用）。
    end
end

% --- Helper function to get 3D coordinates of initial pins and MST-based segments ---
% --- 辅助函数：获取初始引脚的 3D 坐标和基于 MST 的线段 ---
function [nets_to_route, all_pins_info] = get_routing_tasks_from_input(input_data_matrix, design_rule, track_height, layer_info)
% 定义函数 get_routing_tasks_from_input。
    nets_to_route = struct('potential', {}, 'segments', {}); % segments are pairs of indices into all_pins_info
    % 初始化 nets_to_route 结构体数组，包含 'potential' 和 'segments' 字段。
    all_pins_info = struct('id', {}, 'potential', {}, 'original_loc', {}, 'coord_3d', {}); % original_loc = [row_in_input, L]
    % 初始化 all_pins_info 结构体数组，包含引脚 ID、电位、原始位置和 3D 坐标。
    pin_counter = 0;
    % 初始化引脚计数器。

    % 1. Collect all unique pin locations from input_data_matrix
    % 1. 从 input_data_matrix 中收集所有唯一的引脚位置
    value_locations_map = containers.Map('KeyType', 'double', 'ValueType', 'any');
    % 创建一个 Map 来存储每个电位值对应的引脚 ID 列表。
    for r = 1:4 % Iterate through the 4 rows of input_data_matrix
    % 遍历 input_data_matrix 的 4 行。
        for c = 1:size(input_data_matrix, 2)
        % 遍历当前行的所有列。
            val = input_data_matrix(r, c);
            % 获取当前单元格的值。
            if ~isnan(val)
            % 如果值不是 NaN。
                pin_counter = pin_counter + 1;
                % 增加引脚计数器。
                all_pins_info(pin_counter).id = pin_counter;
                % 设置引脚 ID。
                all_pins_info(pin_counter).potential = val;
                % 设置引脚电位。
                all_pins_info(pin_counter).original_loc = [r, c];
                % 设置引脚的原始 2D 位置。
                all_pins_info(pin_counter).coord_3d = map_2d_pin_to_3d_coord([r,c], track_height, layer_info);
                % 调用 map_2d_pin_to_3d_coord 函数将 2D 位置映射到 3D 坐标。
                if ~isKey(value_locations_map, val)
                % 如果 Map 中还没有当前电位。
                    value_locations_map(val) = [];
                    % 初始化该电位对应的引脚 ID 列表。
                end
                % Store the ID of the pin in all_pins_info
                % 将引脚 ID 存储在 all_pins_info 中。
                value_locations_map(val) = [value_locations_map(val), pin_counter];
                % 将当前引脚的 ID 添加到对应电位的列表中。
            end
        end
    end

    % 2. For each potential, run Kruskal to find MST edges
    % 2. 对每个电位，运行 Kruskal 算法以找到 MST 边缘。
    unique_potentials = keys(value_locations_map);
    % 获取 Map 中所有唯一的电位值。
    net_idx_counter = 0;
    % 初始化网络索引计数器。
    for i = 1:length(unique_potentials)
    % 遍历每个唯一的电位。
        potential_val = unique_potentials{i};
        % 获取当前电位值。
        pin_ids_for_potential = value_locations_map(potential_val);
        % 获取当前电位对应的所有引脚 ID。
        if length(pin_ids_for_potential) <= 1
        % 如果当前电位只有一个或没有引脚。
            continue; % No connections needed for single pins
            % 跳过，无需连接。
        end

        % Prepare locations for kruskal_mst_get_edges_wrapper
        % 准备用于 kruskal_mst_get_edges_wrapper 的位置数据。
        % The original kruskal function takes locations as [row_in_input, L_col]
        % 原始的 Kruskal 函数接受的位置是 [input_row_idx, L_col]。
        kruskal_locations = zeros(length(pin_ids_for_potential), 2);
        % 创建一个矩阵来存储 Kruskal 算法所需的 2D 位置。
        for k = 1:length(pin_ids_for_potential)
        % 遍历当前电位的所有引脚 ID。
            kruskal_locations(k,:) = all_pins_info(pin_ids_for_potential(k)).original_loc;
            % 将引脚的原始 2D 位置存储到 kruskal_locations 中。
        end

        % This wrapper should call your kruskal_mst and extract edges
        % 这个包装器应该调用你的 kruskal_mst 并提取边缘。
        % Each edge is [idx1, idx2] referring to rows in kruskal_locations
        % 每个边缘是 [idx1, idx2]，指代 kruskal_locations 中的行索引。
        mst_edges_indices = kruskal_mst_get_edges_wrapper(kruskal_locations, design_rule);
        % 调用 kruskal_mst_get_edges_wrapper 函数获取 MST 边缘。
        if ~isempty(mst_edges_indices)
        % 如果找到了 MST 边缘。
            net_idx_counter = net_idx_counter + 1;
            % 增加网络索引计数器。
            nets_to_route(net_idx_counter).potential = potential_val;
            % 设置当前网络的电位。
            % Convert MST edge indices (relative to kruskal_locations) to global pin IDs
            % 将 MST 边缘索引（相对于 kruskal_locations）转换为全局引脚 ID。
            segments_for_this_net = zeros(size(mst_edges_indices,1), 2);
            % 创建一个矩阵来存储当前网络的线段。
            for k=1:size(mst_edges_indices,1)
            % 遍历每个 MST 边缘。
                segments_for_this_net(k,1) = pin_ids_for_potential(mst_edges_indices(k,1));
                % 将第一个引脚的全局 ID 存储到线段中。
                segments_for_this_net(k,2) = pin_ids_for_potential(mst_edges_indices(k,2));
                % 将第二个引脚的全局 ID 存储到线段中。
            end
            nets_to_route(net_idx_counter).segments = segments_for_this_net;
            % 将线段存储到当前网络中。
        end
    end
end

function coord_3d = map_2d_pin_to_3d_coord(pin_2d, track_height, layer_info)
% 定义函数 map_2d_pin_to_3d_coord，用于将 2D 引脚位置映射到 3D 坐标。
    % pin_2d is [input_row_idx, L_col]
    % pin_2d 是 [input_row_idx, L_col]。
    row_in_input = pin_2d(1);
    % 获取输入数据中的行索引。
    col_in_layer = pin_2d(2);
    % 获取层中的列索引。
    r_in_layer = -1; layer_name_str = '';
    % 初始化层内的行索引和层名称字符串。
    if row_in_input == 1
        r_in_layer = 2; layer_name_str = 'FMD';
        % 如果是 input_data 的第一行，映射到 FMD 层的第 2 行。
    elseif row_in_input == 2
        r_in_layer = 2; layer_name_str = 'BMD';
        % 如果是 input_data 的第二行，映射到 BMD 层的第 2 行。
    elseif row_in_input == 3
        r_in_layer = track_height - 1; layer_name_str = 'FMD';
        % 如果是 input_data 的第三行，映射到 FMD 层的倒数第二行。
    elseif row_in_input == 4
        r_in_layer = track_height - 1; layer_name_str = 'BMD';
        % 如果是 input_data 的第四行，映射到 BMD 层的倒数第二行。
    else
        error('Invalid input_row_idx for pin mapping: %d', row_in_input);
        % 如果输入行索引无效，则抛出错误。
    end
    layer_idx = layer_info.name_to_idx(layer_name_str);
    % 获取层名称对应的索引。
    coord_3d = [r_in_layer, col_in_layer, layer_idx];
    % 返回 3D 坐标 [层内行, 列, 层索引]。
end

% --- KRUSKAL MST related functions (User Provided, with a wrapper) ---
% --- KRUSKAL MST 相关函数（用户提供，带包装器） ---
% Wrapper for kruskal_mst to return edges
% kruskal_mst 的包装器，用于返回边缘。
function mst_edges = kruskal_mst_get_edges_wrapper(locations, design_rule)
% 定义函数 kruskal_mst_get_edges_wrapper，它接受位置和设计规则作为输入。
    num_nodes = size(locations, 1);
    % 获取节点数量。
    mst_edges = [];
    % 初始化 MST 边缘列表。
    if num_nodes <= 1
        return;
        % 如果节点数量小于等于 1，则直接返回。
    end

    edges_data = []; % Store as [u, v, cost, rule_applied_char]
    % 存储边缘数据，格式为 [u, v, 成本, 应用规则的字符]。
    for i = 1:num_nodes
    % 遍历所有节点。
        for j = i+1:num_nodes
        % 遍历剩余的节点（避免重复和自连接）。
            [cost, rule] = code2_get_pair_cost_with_rule(locations(i,:), locations(j,:), design_rule);
            % 调用 code2_get_pair_cost_with_rule 函数计算两个位置之间的成本。
            if isfinite(cost) && cost >= 0
            % 如果成本是有限的且非负。
                edges_data = [edges_data; i, j, cost, rule(1)]; % rule is char
                % 将边缘信息添加到 edges_data 中。
            end
        end
    end
    if isempty(edges_data)
        warning('Kruskal: No valid edges found for locations provided.');
        % 如果没有找到有效边缘，则发出警告。
        return;
    end
    edges_data = sortrows(edges_data, 3); % Sort by cost
    % 根据成本对边缘数据进行排序。
    parent = 1:num_nodes;
    % 初始化并查集的父数组。
    rank = zeros(1, num_nodes);
    % 初始化并查集的秩数组。
    num_edges_in_mst = 0;
    % 初始化 MST 中的边缘数量。
    for k = 1:size(edges_data, 1)
    % 遍历所有排序后的边缘。
        u = edges_data(k, 1);
        v = edges_data(k, 2);
        % cost_val = edges_data(k, 3); % not directly used for edge list
        % 获取当前边缘的两个节点。
        root_u = find_root_kruskal(parent, u);
        root_v = find_root_kruskal(parent, v);
        % 查找两个节点的根。
        if root_u ~= root_v
        % 如果两个节点不在同一个连通分量中。
            mst_edges = [mst_edges; u, v]; % Add edge (indices relative to 'locations')
            % 将边缘添加到 MST 边缘列表中。
            num_edges_in_mst = num_edges_in_mst + 1;
            % 增加 MST 边缘数量。
            % Union sets
            % 合并集合
            if rank(root_u) > rank(root_v)
                parent(root_v) = root_u;
            else
                parent(root_u) = root_v;
                if rank(root_u) == rank(root_v)
                    rank(root_v) = rank(root_v) + 1;
                end
            end
            if num_edges_in_mst == num_nodes - 1
                break; % MST is complete
                % 如果 MST 中的边缘数量达到 num_nodes - 1，则 MST 完成，退出循环。
            end
        end
    end
     % Check if MST actually spans all nodes (if locations formed a single component)
     % 检查 MST 是否实际跨越了所有节点（如果位置形成一个单一组件）。
    if num_nodes > 1 && num_edges_in_mst < num_nodes - 1
        % This can happen if the original points form disconnected groups
        % based on the cost function (e.g. infinite cost between some pairs)
        % The returned 'mst_edges' would be for the largest possible forest.
        % For routing, we might want to connect all points of the same potential regardless of this initial MST cost.
        % However, sticking to the prompt, the MST edges define connections.
        % 这种情况可能发生在原始点根据成本函数形成不连通的组时（例如，某些对之间的成本无限）。
        % 返回的 'mst_edges' 将是最大可能的森林。
        % 对于布线，我们可能希望连接相同电位的所有点，无论这个初始 MST 成本如何。
        % 然而，根据提示，MST 边缘定义了连接。
        % warning('Kruskal: MST for %d nodes only has %d edges. Points might be partitioned by cost function.', num_nodes, num_edges_in_mst);
        % 这是一个注释掉的警告，用于提示 MST 未连接所有点的情况。
    end
end

function root = find_root_kruskal(parent, x) % Kruskal's Union-Find helper
% 定义函数 find_root_kruskal，Kruskal 算法的并查集辅助函数。
    if parent(x) == x
        root = x;
        % 如果 x 是其自身的父节点，则 x 是根。
    else
        parent(x) = find_root_kruskal(parent, parent(x)); % Path compression
        % 路径压缩：将 x 的父节点直接指向根。
        root = parent(x);
        % 返回根节点。
    end
    % Non-recursive version for deep stacks if needed:
    % 如果需要深度堆栈，非递归版本：
    % path = [x];
    % while parent(path(end)) ~= path(end)
    %     path(end+1) = parent(path(end));
    % end
    % root = path(end);
    % for i = 1:length(path)-1
    %     parent(path(i)) = root; % Path compression
    % end
end

% --- User Provided Cost Function (code2_get_pair_cost_with_rule) ---
% --- 用户提供的成本函数 (code2_get_pair_cost_with_rule) ---
% (Copied directly from prompt)
% （直接从提示复制）
function [cost, rule_applied] = code2_get_pair_cost_with_rule(loc1, loc2, design_rule)
% 定义函数 code2_get_pair_cost_with_rule，用于计算两个位置之间的布线成本。
    row1 = loc1(1); L1 = loc1(2);
    row2 = loc2(1); L2 = loc2(2);
    % 获取两个位置的行和列坐标。
    cost = inf; % Default cost if no rule matches
    % 初始化成本为无限大（如果没有任何规则匹配）。
    rule_applied = '';
    % 初始化应用规则的字符串。

    if L1 == L2 % Same L
    % 如果在同一列（L 值相同）。
        if row1 == row2 % This case should ideally not happen if loc1 and loc2 are distinct pins
        % 如果在同一行（理论上不应该发生，因为引脚应该是不同的）。
            cost = design_rule.a; rule_applied = 'a'; % Cost of a point with itself? Or very close.
            % 应用规则 a 的成本。
        elseif (ismember(row1, [1,2]) && ismember(row2, [1,2])) % Both in FMD/BMD top-ish rows
        % 如果都在 FMD/BMD 的顶部行（1 或 2）。
            cost = design_rule.b; rule_applied = 'b';
            % 应用规则 b 的成本。
        elseif (ismember(row1, [3,4]) && ismember(row2, [3,4])) % Both in FMD/BMD bottom-ish rows
        % 如果都在 FMD/BMD 的底部行（3 或 4）。
            cost = design_rule.b; rule_applied = 'b';
            % 应用规则 b 的成本。
        elseif isequal(sort([row1, row2]), [1, 3]) % FMD top to FMD bottom (or vice versa)
        % 如果从 FMD 顶部到 FMD 底部。
            cost = design_rule.c; rule_applied = 'c';
            % 应用规则 c 的成本。
        elseif isequal(sort([row1, row2]), [2, 4]) % BMD top to BMD bottom (or vice versa)
        % 如果从 BMD 顶部到 BMD 底部。
            cost = design_rule.c; rule_applied = 'c';
            % 应用规则 c 的成本。
        else % e.g. [1,2] with [3,4] or [1,4] or [2,3]
        % 其他情况，例如 [1,2] 与 [3,4] 或 [1,4] 或 [2,3]。
            cost = design_rule.d; rule_applied = 'd';
            % 应用规则 d 的成本。
        end
    else % Different L
    % 如果在不同列（L 值不同）。
        dL = abs(L1 - L2);
        % 计算 L 之间的距离。
        rpair = sort([row1, row2]);
        % 对行进行排序。
        if row1 == row2 % Same row, different L
        % 如果在同一行，但不同列。
            cost = design_rule.e1 + design_rule.e2 * dL; rule_applied = 'e';
            % 应用规则 e 的成本。
        elseif isequal(rpair, [1, 2]) % FMD row with BMD row (same side, top)
        % 如果是 FMD 行与 BMD 行（同一侧，顶部）。
            cost = design_rule.f1 + design_rule.f2 * dL; rule_applied = 'f';
            % 应用规则 f 的成本。
        elseif isequal(rpair, [3, 4]) % FMD row with BMD row (same side, bottom)
        % 如果是 FMD 行与 BMD 行（同一侧，底部）。
            cost = design_rule.f1 + design_rule.f2 * dL; rule_applied = 'f';
            % 应用规则 f 的成本。
        elseif isequal(rpair, [1, 3]) % FMD top with FMD bottom (different L)
        % 如果是 FMD 顶部与 FMD 底部（不同 L）。
            cost = design_rule.g1 + design_rule.g2 * dL; rule_applied = 'g';
            % 应用规则 g 的成本。
        elseif isequal(rpair, [2, 4]) % BMD top with BMD bottom (different L)
        % 如果是 BMD 顶部与 BMD 底部（不同 L）。
            cost = design_rule.g1 + design_rule.g2 * dL; rule_applied = 'g';
            % 应用规则 g 的成本。
        elseif isequal(rpair, [2, 3]) % BMD top with FMD bottom (diagonal across F/B)
        % 如果是 BMD 顶部与 FMD 底部（跨越 F/B 的对角线）。
            cost = design_rule.h1 + design_rule.h2 * dL; rule_applied = 'h';
            % 应用规则 h 的成本。
        elseif isequal(rpair, [1, 4]) % FMD top with BMD bottom (diagonal across F/B)
        % 如果是 FMD 顶部与 BMD 底部（跨越 F/B 的对角线）。
            cost = design_rule.h1 + design_rule.h2 * dL; rule_applied = 'h';
            % 应用规则 h 的成本。
        end
    end
    % if isinf(cost)
    %     warning('Infinite cost between loc (%d,%d) and (%d,%d)', row1, L1, row2, L2);
    % end
    % 这是一个注释掉的警告，用于提示成本为无限大的情况。
end

% --- A* Pathfinder ---
% --- A* 路径查找器 ---
function [path_coords, cost] = pathfinder_astar(layout, start_coord_3d, end_coord_3d, potential, layer_info, L_max, H_max)
% 定义函数 pathfinder_astar，用于执行 A* 路径查找。
    % start_coord_3d, end_coord_3d: [r, c, layer_idx]
    % start_coord_3d, end_coord_3d: [行, 列, 层索引]。
    % potential: the numeric value to route
    % potential: 要布线的数值电位。
    % layer_info: struct with names, types, etc.
    % layer_info: 包含名称、类型等的结构体。
    % L_max, H_max: dimensions of layers
    % L_max, H_max: 层的尺寸。
    path_coords = [];
    % 初始化路径坐标。
    cost = inf;
    % 初始化成本为无限大。
    start_node.g = 0;
    % 起始节点的 g 值（从起始点到当前点的成本）为 0。
    start_node.h = heuristic(start_coord_3d, end_coord_3d, layer_info);
    % 起始节点的 h 值（启发式估算从当前点到目标点的成本）。
    start_node.f = start_node.g + start_node.h;
    % 起始节点的 f 值（g + h）。
    start_node.coord = start_coord_3d;
    % 起始节点的坐标。
    start_node.parent_coord = []; % No parent for start
    % 起始节点没有父节点。
    pq = PriorityQueue(); % Needs a simple priority queue implementation
    % 创建一个优先队列（需要一个简单的优先队列实现）。
    pq.insert(start_node, start_node.f);
    % 将起始节点插入优先队列。
    closed_set = containers.Map('KeyType','char','ValueType','any'); % Store node coords (as string) that have been processed
    % 创建一个 Map 来存储已处理过的节点（以字符串形式存储坐标）。

    while ~pq.isEmpty()
    % 当优先队列不为空时循环。
        current_node_pq = pq.pop(); % Gets the element, not the priority
        % 从优先队列中弹出 f 值最小的节点。
        current_coord_str = mat2str(current_node_pq.coord);
        % 将当前节点的坐标转换为字符串。
        if isKey(closed_set, current_coord_str)
        % 如果当前节点已在 closed_set 中（已处理过）。
            continue;
            % 跳过，处理下一个节点。
        end
        closed_set(current_coord_str) = current_node_pq;
        % 将当前节点添加到 closed_set 中。

        if all(current_node_pq.coord == end_coord_3d) % Goal reached
        % 如果当前节点是目标节点。
            path_coords = reconstruct_path(closed_set, current_node_pq.coord);
            % 重构路径。
            cost = current_node_pq.f;
            % 路径成本为当前节点的 f 值。
            return;
            % 返回路径和成本。
        end

        neighbors = get_valid_neighbors(layout, current_node_pq.coord, potential, layer_info, L_max, H_max);
        % 获取当前节点的有效邻居。
        for i = 1:size(neighbors, 1)
        % 遍历每个邻居。
            neighbor_coord = neighbors(i,:);
            % 获取邻居的坐标。
            neighbor_coord_str = mat2str(neighbor_coord);
            % 将邻居坐标转换为字符串。
            if isKey(closed_set, neighbor_coord_str) % Already processed this neighbor fully
            % 如果邻居已在 closed_set 中（已完全处理过）。
                continue; % Skip fully processed nodes
            end
            
            % Check if neighbor is blocked by another potential or -2
            % 检查邻居是否被其他电位或 -2 阻塞。
            % This check should also be inside get_valid_neighbors, but double check
            % 这个检查也应该在 get_valid_neighbors 内部，但这里进行双重检查。
            n_r = neighbor_coord(1); n_c = neighbor_coord(2); n_l_idx = neighbor_coord(3);
            % 获取邻居的行、列和层索引。
            layer_name = layer_info.idx_to_name(n_l_idx);
            % 获取邻居所在层的名称。
            if layout.(layer_name)(n_r, n_c) ~= -1 && layout.(layer_name)(n_r, n_c) ~= potential
            % 如果邻居单元格不是空的 (-1) 且不属于当前电位。
                continue; % Blocked by other potential
                % 跳过，被其他电位阻塞。
            end
            if layout.(layer_name)(n_r, n_c) == -2
            % 如果邻居单元格是不可用的 (-2)。
                 continue; % Blocked by -2
                 % 跳过，被 -2 阻塞。
            end

            g_score = current_node_pq.g + 1; % Simple cost: 1 per step (vias could be more)
                                           % 简单的成本：每步 1（过孔可以更高）。
                                           % Vias often have higher cost, adjust '1' based on neighbor type
                                           % 过孔通常成本更高，根据邻居类型调整 '1'。
            
            % Check if neighbor is already in PQ with a higher g_score
            % 检查邻居是否已在 PQ 中且具有更高的 g_score。
            % (This part makes it a bit like Dijkstra if not careful with PQ updates,
            % 但对于 A* 来说，它是否在 closed_set 中或者是否有更好的路径。）
            existing_in_pq = pq.find(neighbor_coord); % Needs a find method in PQ by coord
            % 查找优先队列中是否存在该邻居。
            
            if isempty(existing_in_pq) || g_score < existing_in_pq.g
            % 如果邻居不在优先队列中，或者通过当前路径到达邻居的 g_score 更低。
                neighbor_node.g = g_score;
                % 更新邻居的 g 值。
                neighbor_node.h = heuristic(neighbor_coord, end_coord_3d, layer_info);
                % 计算邻居的 h 值。
                neighbor_node.f = neighbor_node.g + neighbor_node.h;
                % 计算邻居的 f 值。
                neighbor_node.coord = neighbor_coord;
                % 设置邻居的坐标。
                neighbor_node.parent_coord = current_node_pq.coord;
                % 设置邻居的父节点。
                
                if ~isempty(existing_in_pq)
                    pq.remove(existing_in_pq); % If PQ supports removal/update
                    % 如果优先队列中存在该邻居，且新的路径更好，则移除旧的。
                end
                pq.insert(neighbor_node, neighbor_node.f);
                % 将邻居插入优先队列。
            end
        end
    end
end

function h_val = heuristic(coord1, coord2, layer_info)
% 定义启发式函数 heuristic。
    % Manhattan distance, slightly weighted by layer difference
    % 曼哈顿距离，稍微加权层差异。
    dr = abs(coord1(1) - coord2(1));
    % 行方向的距离。
    dc = abs(coord1(2) - coord2(2));
    % 列方向的距离。
    dl = abs(coord1(3) - coord2(3)); % Difference in layer indices
    % 层索引的差异。
    
    % Basic heuristic: sum of differences.
    % 基本启发式：差异之和。
    % More advanced: consider number of vias needed based on layer types.
    % 更高级：根据层类型考虑所需的过孔数量。
    % If layers are far apart and of same type (e.g. H to H), vias are necessary.
    % 如果层相距较远且类型相同（例如 H 到 H），则过孔是必需的。
    h_val = dr + dc + dl * 2; % Penalize layer changes a bit more
    % 计算启发式值，层变化有更高的惩罚。
end

function neighbors = get_valid_neighbors(layout, current_coord, potential, layer_info, L_max, H_max)
% 定义函数 get_valid_neighbors，获取当前坐标的有效邻居。
    % current_coord: [r, c, layer_idx]
    % current_coord: [行, 列, 层索引]。
    r = current_coord(1); c = current_coord(2); l_idx = current_coord(3);
    % 获取当前坐标的行、列和层索引。
    current_layer_name = layer_info.idx_to_name(l_idx);
    % 获取当前层的名称。
    current_layer_type = layer_info.types{l_idx};
    % 获取当前层的类型。
    neighbors = zeros(0,3); % Initialize as empty
    % 初始化邻居列表为空。

    % --- Intra-layer connections ---
    % --- 层内连接 ---
    if strcmp(current_layer_type, 'H') % Horizontal layers: FM0, BM0, FM2, BM2
    % 如果是水平金属层。
        moves = [0 -1; 0 1]; % Left, Right
        % 定义左右移动。
    elseif strcmp(current_layer_type, 'V') % Vertical layers: FMD, BMD, FM1, BM1
    % 如果是垂直金属层。
        moves = [-1 0; 1 0]; % Up, Down
        % 定义上下移动。
    else % VIA layer - no intra-layer routing movement in this model
        moves = [];
        % 过孔层没有层内移动。
    end

    for i=1:size(moves,1)
    % 遍历可能的移动。
        nr = r + moves(i,1);
        nc = c + moves(i,2);
        % 计算新坐标。
        if nr >= 1 && nr <= H_max && nc >= 1 && nc <= L_max
        % 如果新坐标在布局范围内。
            % Cell must be -1 (empty) or same potential (part of the net already)
            % 单元格必须是 -1（空的）或与当前电位相同（已是网络的一部分）。
            if layout.(current_layer_name)(nr,nc) == -1 || layout.(current_layer_name)(nr,nc) == potential
            % 如果目标单元格是空的或属于当前电位。
                neighbors(end+1,:) = [nr, nc, l_idx];
                % 添加为邻居。
            end
        end
    end

    % --- Inter-layer connections (Vias) ---
    % --- 层间连接（过孔） ---
    % 1. From a Metal layer TO an adjacent Metal layer THROUGH a VIA layer
    % 1. 从一个金属层通过一个过孔层连接到相邻的金属层。
    if strcmp(current_layer_type, 'H') || strcmp(current_layer_type, 'V')
    % 如果是金属层。
        [via_names_up, metal_names_up, via_names_down, metal_names_down] = get_adjacent_vias_metals(l_idx, layer_info);
        % 获取相邻过孔和金属层的名称。
        for i=1:numel(via_names_up) % Connections "upwards" or to next layer in sequence
        % 遍历“向上”连接。
            via_l_idx = layer_info.name_to_idx(via_names_up{i});
            adj_metal_l_idx = layer_info.name_to_idx(metal_names_up{i});
            % 获取过孔层和相邻金属层的索引。
            % Check if via cell and target metal cell are usable
            % 检查过孔单元格和目标金属单元格是否可用。
            if (layout.(via_names_up{i})(r,c) == -1 || layout.(via_names_up{i})(r,c) == potential) && ...
               (layout.(metal_names_up{i})(r,c) == -1 || layout.(metal_names_up{i})(r,c) == potential)
            % 如果过孔单元格和相邻金属单元格是空的或属于当前电位。
                % The "neighbor" is the cell in the adjacent metal layer. The path implies using the via.
                % “邻居”是相邻金属层中的单元格。路径意味着使用过孔。
                neighbors(end+1,:) = [r, c, adj_metal_l_idx];
                % 添加相邻金属层中的单元格作为邻居。
            end
        end
        for i=1:numel(via_names_down) % Connections "downwards"
        % 遍历“向下”连接。
            via_l_idx = layer_info.name_to_idx(via_names_down{i});
            adj_metal_l_idx = layer_info.name_to_idx(metal_names_down{i});
            if (layout.(via_names_down{i})(r,c) == -1 || layout.(via_names_down{i})(r,c) == potential) && ...
               (layout.(metal_names_down{i})(r,c) == -1 || layout.(metal_names_down{i})(r,c) == potential)
                neighbors(end+1,:) = [r, c, adj_metal_l_idx];
                % 添加相邻金属层中的单元格作为邻居。
            end
        end
    end
    
    % V_FMD_BMD special handling: It connects FMD and BMD at the same (r,c)
    % V_FMD_BMD 特殊处理：它在相同的 (r,c) 处连接 FMD 和 BMD。
    if strcmp(current_layer_name, 'FMD')
    % 如果当前层是 FMD。
        via_name = 'V_FMD_BMD'; adj_metal_name = 'BMD';
        adj_metal_l_idx = layer_info.name_to_idx(adj_metal_name);
        % Check if the V_FMD_BMD cell itself is usable (-1 or current potential)
        % 检查 V_FMD_BMD 单元格本身是否可用（-1 或当前电位）。
        % The problem states V_FMD_BMD(2:H-1,:) are -2. So this via is only usable at row 1 and H.
        % 问题说明 V_FMD_BMD(2:H-1,:) 是 -2。因此这个过孔只能在第 1 行和 H 行使用。
        if (layout.(via_name)(r,c) == -1 || layout.(via_name)(r,c) == potential) && ...
           (layout.(adj_metal_name)(r,c) == -1 || layout.(adj_metal_name)(r,c) == potential)
             neighbors(end+1,:) = [r, c, adj_metal_l_idx];
             % 添加 BMD 层中的单元格作为邻居。
        end
    elseif strcmp(current_layer_name, 'BMD')
    % 如果当前层是 BMD。
        via_name = 'V_FMD_BMD'; adj_metal_name = 'FMD';
        adj_metal_l_idx = layer_info.name_to_idx(adj_metal_name);
         if (layout.(via_name)(r,c) == -1 || layout.(via_name)(r,c) == potential) && ...
           (layout.(adj_metal_name)(r,c) == -1 || layout.(adj_metal_name)(r,c) == potential)
             neighbors(end+1,:) = [r, c, adj_metal_l_idx];
             % 添加 FMD 层中的单元格作为邻居。
        end
    end
end

function [via_names_up, metal_names_up, via_names_down, metal_names_down] = get_adjacent_vias_metals(current_metal_layer_idx, layer_info)
% 定义函数 get_adjacent_vias_metals，获取相邻的过孔和金属层名称。
    % For a given metal layer index, find directly connectable vias and the metals on their other side.
    % 对于给定的金属层索引，查找可直接连接的过孔及其另一侧的金属层。
    % "Up" usually means towards FM2 or BM2 (higher index in typical layout stack order if sorted).
    % “向上”通常指 FM2 或 BM2（如果在典型布局堆栈顺序中排序，则索引更高）。
    % "Down" usually means towards FMD or BMD.
    % “向下”通常指 FMD 或 BMD。
    % This depends on the exact ordering in layer_info.names. Let's use sequence.
    % 这取决于 layer_info.names 中的精确顺序。我们使用序列。
    via_names_up = {}; metal_names_up = {};
    via_names_down = {}; metal_names_down = {};
    current_metal_name = layer_info.idx_to_name(current_metal_layer_idx);
    % 获取当前金属层的名称。

    % Check layer before current_metal_layer_idx (potential via "down")
    % 检查当前金属层索引之前的层（潜在的“向下”过孔）。
    if current_metal_layer_idx > 1
    % 如果当前金属层不是第一层。
        prev_layer_idx = current_metal_layer_idx - 1;
        prev_layer_name = layer_info.idx_to_name(prev_layer_idx);
        % 获取前一层的索引和名称。
        if strcmp(layer_info.types{prev_layer_idx}, 'VIA') % If it's a VIA
        % 如果前一层是过孔层。
            % Check if this VIA connects to current_metal_name
            % 检查此过孔是否连接到 current_metal_name。
            [m1, m2] = get_metals_for_via(prev_layer_name, layer_info);
            % 获取过孔连接的两个金属层名称。
            if strcmp(m1, current_metal_name) && ~isempty(m2)
                via_names_down{end+1} = prev_layer_name;
                metal_names_down{end+1} = m2;
            elseif strcmp(m2, current_metal_name) && ~isempty(m1)
                via_names_down{end+1} = prev_layer_name;
                metal_names_down{end+1} = m1;
            end
        end
    end

    % Check layer after current_metal_layer_idx (potential via "up")
    % 检查当前金属层索引之后的层（潜在的“向上”过孔）。
    if current_metal_layer_idx < numel(layer_info.names)
    % 如果当前金属层不是最后一层。
        next_layer_idx = current_metal_layer_idx + 1;
        next_layer_name = layer_info.idx_to_name(next_layer_idx);
        % 获取下一层的索引和名称。
        if strcmp(layer_info.types{next_layer_idx}, 'VIA') % If it's a VIA
        % 如果下一层是过孔层。
            [m1, m2] = get_metals_for_via(next_layer_name, layer_info);
            if strcmp(m1, current_metal_name) && ~isempty(m2)
                via_names_up{end+1} = next_layer_name;
                metal_names_up{end+1} = m2;
            elseif strcmp(m2, current_metal_name) && ~isempty(m1)
                via_names_up{end+1} = next_layer_name;
                metal_names_up{end+1} = m1;
            end
        end
    end
end

function [metal1_name, metal2_name] = get_metals_for_via(via_layer_name, layer_info)
% 定义函数 get_metals_for_via，给定过孔层名称，找到它连接的两个金属层。
    metal1_name = ''; metal2_name = '';
    % 初始化金属层名称。
    parts = strsplit(via_layer_name, '_'); % e.g., FV_M1_M2 or V_FMD_BMD
    % 根据 '_' 分割过孔层名称。
    if numel(parts) < 3
        % warning('Via name %s does not follow expected format.', via_layer_name);
        return;
        % 如果分割后的部分少于 3 个，则返回。
    end
    
    prefix = parts{1};
    p2 = parts{2};
    p3 = parts{3};
    if strcmp(prefix, 'FV') % e.g., FV_M1_M2 connects FM1 and FM2
        metal1_name = ['F' p2];
        metal2_name = ['F' p3];
        % 如果前缀是 FV，则连接 F 前缀的两个金属层。
    elseif strcmp(prefix, 'BV') % e.g., BV_M0_M1 connects BM0 and BM1
        metal1_name = ['B' p2];
        metal2_name = ['B' p3];
        % 如果前缀是 BV，则连接 B 前缀的两个金属层。
    elseif strcmp(prefix, 'V') && strcmp(p2,'FMD') && strcmp(p3,'BMD') % V_FMD_BMD
        metal1_name = 'FMD';
        metal2_name = 'BMD';
        % 如果是 V_FMD_BMD，则连接 FMD 和 BMD。
    else
        % warning('Unknown via naming convention for %s', via_layer_name);
        % 如果是未知的过孔命名约定，则发出警告。
    end
    
    % Validate that these metal names exist in layer_info
    % 验证这些金属名称是否存在于 layer_info 中。
    if ~isempty(metal1_name) && ~isKey(layer_info.name_to_idx, metal1_name)
        % warning('Metal %s (from via %s) not found in layer_info.', metal1_name, via_layer_name);
        metal1_name = '';
    end
    if ~isempty(metal2_name) && ~isKey(layer_info.name_to_idx, metal2_name)
        % warning('Metal %s (from via %s) not found in layer_info.', metal2_name, via_layer_name);
        metal2_name = '';
    end
end

function path = reconstruct_path(closed_set, current_coord)
% 定义函数 reconstruct_path，根据 closed_set 和当前坐标重构路径。
    path = current_coord;
    % 初始化路径为当前坐标。
    current_coord_str = mat2str(current_coord);
    % 将当前坐标转换为字符串。
    while ~isempty(closed_set(current_coord_str).parent_coord)
    % 当当前节点的父节点不为空时循环。
        parent_c = closed_set(current_coord_str).parent_coord;
        % 获取父节点坐标。
        path = [parent_c; path];
        % 将父节点添加到路径的开头。
        current_coord_str = mat2str(parent_c);
        % 更新当前坐标为父节点坐标。
        if ~isKey(closed_set, current_coord_str) % Should not happen if path reconstruction is correct
        % 如果父节点不在 closed_set 中（理论上不应该发生）。
            error('Error in path reconstruction: parent not in closed set.');
            % 抛出路径重构错误。
        end
    end
end

% --- Apply Path to Layout ---
% --- 将路径应用到布局 ---
function [layout_out, success] = apply_path_to_layout(layout_in, path_coords, potential, layer_info)
% 定义函数 apply_path_to_layout，将路径应用到布局上。
    layout_out = layout_in;
    % 复制输入布局到输出布局。
    success = true;
    % 初始化成功标志为 true。
    % path_coords is N x 3, where each row is [r, c, layer_idx]
    % path_coords 是 N x 3 矩阵，每行是 [行, 列, 层索引]。
    
    % First, check for conflicts along the path before applying
    % 首先，在应用路径之前检查路径上的冲突。
    for i = 1:size(path_coords,1)
    % 遍历路径中的每个点。
        pt = path_coords(i,:);
        % 获取当前点。
        r = pt(1); c = pt(2); l_idx = pt(3);
        % 获取点的行、列和层索引。
        layer_name = layer_info.idx_to_name(l_idx);
        % 获取层名称。
        if layout_out.(layer_name)(r,c) ~= -1 && layout_out.(layer_name)(r,c) ~= potential
        % 如果单元格不是空的且不属于当前电位。
            success = false; % Conflict with another potential
            % 设置成功标志为 false，表示与其他电位冲突。
            % disp(['Conflict applying path at (', num2str(r), ',', num2str(c), ') in layer ', layer_name, '. Expected -1 or ', num2str(potential), ', found ', num2str(layout_out.(layer_name)(r,c))]);
            return; % Do not apply if conflict
            % 返回，不应用路径。
        end
    end

    % Apply path if no conflicts found in the check
    % 如果检查中没有发现冲突，则应用路径。
    for i = 1:size(path_coords,1)
    % 遍历路径中的每个点。
        pt = path_coords(i,:);
        r = pt(1); c = pt(2); l_idx = pt(3);
        layer_name = layer_info.idx_to_name(l_idx);
        layout_out.(layer_name)(r,c) = potential;
        % 将路径上的单元格设置为当前电位。
        
        % If this point implies a via was used, mark the via layer too.
        % 如果此点意味着使用了过孔，也标记过孔层。
        % The path_coords should ideally contain explicit via cells if they are separate entities.
        % 理想情况下，如果过孔是独立实体，path_coords 应该包含明确的过孔单元格。
        % Our current get_valid_neighbors implies via usage by jumping layers.
        % 我们当前的 get_valid_neighbors 通过跳层来暗示过孔的使用。
        % We need to mark the via cell between path_coords(i-1) and path_coords(i) if it's a layer jump.
        % 如果是层跳变，我们需要标记 path_coords(i-1) 和 path_coords(i) 之间的过孔单元格。
        if i > 1
            prev_pt = path_coords(i-1,:);
            pr = prev_pt(1); pc = prev_pt(2); pl_idx = prev_pt(3);
            % 获取前一个点的坐标。
            % If current point (r,c,l_idx) and previous point (pr,pc,pl_idx)
            % are at same (r,c) but different l_idx, a via was traversed.
            % 如果当前点和前一个点在相同的 (r,c) 但不同的 l_idx，则表示经过了一个过孔。
            if r == pr && c == pc && l_idx ~= pl_idx
                % Find the via layer between l_idx and pl_idx
                % 找到 l_idx 和 pl_idx 之间的过孔层。
                via_layer_l_idx = find_via_layer_between(l_idx, pl_idx, layer_info);
                if via_layer_l_idx > 0
                    via_layer_name = layer_info.idx_to_name(via_layer_l_idx);
                    if layout_out.(via_layer_name)(r,c) ~= -1 && layout_out.(via_layer_name)(r,c) ~= potential
                         success = false; % conflict on via
                         % 如果过孔上发生冲突，理论上 A* 应该已经检查过。
                         % For safety, revert (or don't apply from start if pre-checked)
                         % 为了安全，回滚（或者如果已预先检查，则不从头开始应用）。
                         % This part needs robust handling. For now, assume A* path is clear.
                         % 这部分需要健壮的处理。目前，假设 A* 路径是清晰的。
                         return;
                    end
                    layout_out.(via_layer_name)(r,c) = potential;
                    % 标记过孔层。
                end
            end
        end
    end
end

function via_l_idx = find_via_layer_between(metal1_l_idx, metal2_l_idx, layer_info)
% 定义函数 find_via_layer_between，查找两个金属层之间的过孔层索引。
    via_l_idx = 0; % Not found
    % 初始化过孔层索引为 0（未找到）。
    % Check layers between metal1_l_idx and metal2_l_idx in sequence
    % 按顺序检查 metal1_l_idx 和 metal2_l_idx 之间的层。
    idx_start = min(metal1_l_idx, metal2_l_idx);
    idx_end = max(metal1_l_idx, metal2_l_idx);
    if abs(metal1_l_idx - metal2_l_idx) ~= 2 && ~( (strcmp(layer_info.idx_to_name(metal1_l_idx),'FMD') && strcmp(layer_info.idx_to_name(metal2_l_idx),'BMD')) || ...
                                                 (strcmp(layer_info.idx_to_name(metal2_l_idx),'FMD') && strcmp(layer_info.idx_to_name(metal1_l_idx),'BMD')) )
        % Typically, metal layers are separated by one via layer in sequence
        % 通常，金属层在序列中由一个过孔层隔开。
        % Special case for V_FMD_BMD which might not be in direct sequence
        % V_FMD_BMD 的特殊情况可能不在直接序列中。
         if (strcmp(layer_info.idx_to_name(metal1_l_idx),'FMD') && strcmp(layer_info.idx_to_name(metal2_l_idx),'BMD')) || ...
            (strcmp(layer_info.idx_to_name(metal2_l_idx),'FMD') && strcmp(layer_info.idx_to_name(metal1_l_idx),'BMD'))
            via_l_idx = layer_info.name_to_idx('V_FMD_BMD');
            return;
         end
        return; % Not adjacent through a single via in sequence
        % 如果不是通过序列中的单个过孔相邻，则返回。
    end
    
    potential_via_idx = (idx_start + idx_end) / 2;
    % 计算潜在的过孔索引。
    if round(potential_via_idx) == potential_via_idx && strcmp(layer_info.types{potential_via_idx}, 'VIA')
    % 如果潜在过孔索引是整数且该层是过孔层。
        % Check if this via connects these specific metals
        % 检查此过孔是否连接这些特定的金属层。
        via_name = layer_info.idx_to_name(potential_via_idx);
        [m1_conn, m2_conn] = get_metals_for_via(via_name, layer_info);
        metal1_name_actual = layer_info.idx_to_name(metal1_l_idx);
        metal2_name_actual = layer_info.idx_to_name(metal2_l_idx);
        if (strcmp(m1_conn, metal1_name_actual) && strcmp(m2_conn, metal2_name_actual)) || ...
           (strcmp(m1_conn, metal2_name_actual) && strcmp(m2_conn, metal1_name_actual))
            via_l_idx = potential_via_idx;
            return;
        end
    end
    % Fallback for V_FMD_BMD if not caught by sequence
    % 如果未被序列捕获，则为 V_FMD_BMD 的回退。
    if (strcmp(layer_info.idx_to_name(metal1_l_idx),'FMD') && strcmp(layer_info.idx_to_name(metal2_l_idx),'BMD')) || ...
       (strcmp(layer_info.idx_to_name(metal2_l_idx),'FMD') && strcmp(layer_info.idx_to_name(metal1_l_idx),'BMD'))
        via_l_idx = layer_info.name_to_idx('V_FMD_BMD');
    end
end

% --- Rip-up Path from Layout (Simplified) ---
% --- 从布局中拆除路径（简化版） ---
function layout_out = remove_path_from_layout(layout_in, path_coords, layer_info)
% 定义函数 remove_path_from_layout，从布局中移除路径。
    % path_coords is N x 3 from a previously routed segment
    % path_coords 是来自先前布线线段的 N x 3 矩阵。
    layout_out = layout_in;
    % 复制输入布局到输出布局。
    if isempty(path_coords)
        return;
    end
    for i = 1:size(path_coords,1)
        pt = path_coords(i,:);
        r = pt(1); c = pt(2); l_idx = pt(3);
        layer_name = layer_info.idx_to_name(l_idx);
        % Only set to -1 if it currently holds the potential of the path being removed.
        % 仅当它当前持有要移除的路径的电位时才设置为 -1。
        % This is tricky if multiple paths of same potential overlap.
        % 如果多个相同电位的路径重叠，这将很棘手。
        % For rip-up, assume we are removing this specific instance.
        % 对于拆除，假设我们正在移除这个特定的实例。
        layout_out.(layer_name)(r,c) = -1; % Set back to unused
        % 将单元格设置回 -1（未使用）。
        % Also clear implied vias
        % 也清除隐含的过孔。
        if i > 1
            prev_pt = path_coords(i-1,:);
            pr = prev_pt(1); pc = prev_pt(2); pl_idx = prev_pt(3);
            if r == pr && c == pc && l_idx ~= pl_idx
                via_layer_l_idx = find_via_layer_between(l_idx, pl_idx, layer_info);
                if via_layer_l_idx > 0
                    via_layer_name = layer_info.idx_to_name(via_layer_l_idx);
                    layout_out.(via_layer_name)(r,c) = -1;
                    % 清除过孔层。
                end
            end
        end
    end
end

% --- Connectivity Check (BFS-based) ---
% --- 连接性检查（基于 BFS） ---
function [is_connected, component_count, total_potential_cells] = check_potential_connectivity_bfs(layout, potential_value, layer_info)
% 定义函数 check_potential_connectivity_bfs，使用 BFS 检查电位的连接性。
    H_max = layer_info.track_height;
    L_max = layer_info.L;
    
    q = java.util.LinkedList();
    % 创建一个 Java LinkedList 作为队列。
    visited_map = containers.Map('KeyType','char','ValueType','logical');
    % 创建一个 Map 来存储已访问的单元格。
    
    first_point_found = false;
    start_coord = [];
    total_potential_cells = 0;
    % Find all cells with the potential_value and count them / find a starting point
    % 找到所有具有 potential_value 的单元格并计数 / 找到一个起始点。
    for l_idx = 1:numel(layer_info.names)
    % 遍历所有层。
        layer_name = layer_info.idx_to_name(l_idx);
        [rows, cols] = find(layout.(layer_name) == potential_value);
        % 查找当前层中所有具有 potential_value 的单元格。
        for k=1:numel(rows)
        % 遍历每个找到的单元格。
            total_potential_cells = total_potential_cells + 1;
            % 增加总潜在单元格数量。
            if ~first_point_found
            % 如果还没有找到起始点。
                start_coord = [rows(k), cols(k), l_idx];
                q.add(start_coord);
                visited_map(mat2str(start_coord)) = true;
                first_point_found = true;
                % 设置起始点，添加到队列并标记为已访问。
            end
        end
    end
    if total_potential_cells == 0 || total_potential_cells == 1
    % 如果总潜在单元格数量为 0 或 1。
        is_connected = true;
        component_count = total_potential_cells;
        return;
        % 视为已连接，返回。
    end
    
    if ~first_point_found % Should not happen if total_potential_cells > 0
    % 如果没有找到起始点（理论上不应该发生，因为 total_potential_cells > 0）。
        is_connected = false; % Or true if 0 cells
        component_count = 0;
        return;
        % 返回未连接。
    end

    component_count = 0;
    while ~q.isEmpty()
    % 当队列不为空时循环。
        current_coord = q.remove();
        % 从队列中移除当前坐标。
        component_count = component_count + 1;
        % 增加连通分量中的单元格数量。
        
        % Get neighbors (similar to A* but only on same potential)
        % 获取邻居（类似于 A* 但只针对相同电位的单元格）。
        neighbors = get_connected_neighbors_for_check(layout, current_coord, potential_value, layer_info, L_max, H_max);
        
        for i=1:size(neighbors,1)
        % 遍历每个邻居。
            neighbor_coord = neighbors(i,:);
            neighbor_coord_str = mat2str(neighbor_coord);
            if ~isKey(visited_map, neighbor_coord_str)
            % 如果邻居尚未被访问。
                % Check if the neighbor actually has the potential value
                % (get_connected_neighbors_for_check should ensure this)
                % 检查邻居是否实际具有电位值（get_connected_neighbors_for_check 应该确保这一点）。
                 n_r = neighbor_coord(1); n_c = neighbor_coord(2); n_l_idx = neighbor_coord(3);
                 n_layer_name = layer_info.idx_to_name(n_l_idx);
                 if layout.(n_layer_name)(n_r,n_c) == potential_value
                    visited_map(neighbor_coord_str) = true;
                    q.add(neighbor_coord);
                 end
            end
        end
    end
    is_connected = (component_count == total_potential_cells);
    % 如果连通分量中的单元格数量等于总潜在单元格数量，则视为已连接。
end

function neighbors = get_connected_neighbors_for_check(layout, current_coord, potential, layer_info, L_max, H_max)
% 定义函数 get_connected_neighbors_for_check，获取连接性检查中合格的邻居。
    % Similar to get_valid_neighbors but for connectivity check (must be same potential)
    % 类似于 get_valid_neighbors，但用于连接性检查（必须是相同电位）。
    r = current_coord(1); c = current_coord(2); l_idx = current_coord(3);
    current_layer_name = layer_info.idx_to_name(l_idx);
    current_layer_type = layer_info.types{l_idx};
    neighbors = zeros(0,3);
    % Intra-layer
    % 层内。
    if strcmp(current_layer_type, 'H')
        moves = [0 -1; 0 1];
    elseif strcmp(current_layer_type, 'V')
        moves = [-1 0; 1 0];
    else % VIA
        moves = [];
    end
    for i=1:size(moves,1)
        nr = r + moves(i,1); nc = c + moves(i,2);
        if nr >= 1 && nr <= H_max && nc >= 1 && nc <= L_max && layout.(current_layer_name)(nr,nc) == potential
            neighbors(end+1,:) = [nr, nc, l_idx];
        end
    end

    % Inter-layer (through Vias that also have the same potential)
    % 层间（通过也具有相同电位的过孔）。
    if strcmp(current_layer_type, 'H') || strcmp(current_layer_type, 'V') % From Metal
        [via_names_up, metal_names_up, via_names_down, metal_names_down] = get_adjacent_vias_metals(l_idx, layer_info);
        for i=1:numel(via_names_up)
            adj_metal_l_idx = layer_info.name_to_idx(metal_names_up{i});
            if layout.(via_names_up{i})(r,c) == potential && layout.(metal_names_up{i})(r,c) == potential
                neighbors(end+1,:) = [r, c, adj_metal_l_idx];
            end
        end
        for i=1:numel(via_names_down)
            adj_metal_l_idx = layer_info.name_to_idx(metal_names_down{i});
            if layout.(via_names_down{i})(r,c) == potential && layout.(via_names_down{i})(r,c) == potential
                neighbors(end+1,:) = [r, c, adj_metal_l_idx];
            end
        end
    end
    % V_FMD_BMD specific check
    % V_FMD_BMD 特殊检查。
    if strcmp(current_layer_name, 'FMD') && layout.V_FMD_BMD(r,c) == potential && layout.BMD(r,c) == potential
        neighbors(end+1,:) = [r, c, layer_info.name_to_idx('BMD')];
    elseif strcmp(current_layer_name, 'BMD') && layout.V_FMD_BMD(r,c) == potential && layout.FMD(r,c) == potential
        neighbors(end+1,:) = [r, c, layer_info.name_to_idx('FMD')];
    end
end

% --- Visualization ---
% --- 可视化 ---
function visualize_layout(layout, layer_info, unique_potentials)
% 定义函数 visualize_layout，用于可视化布局。
    num_layers = numel(layer_info.names);
    
    % Create a colormap
    % 创建一个颜色图。
    % Max potential value for colormap scaling. Add 2 for -1 and -2.
    % 用于颜色图缩放的最大电位值。为 -1 和 -2 额外加 2。
    max_potential_val = 0;
    if ~isempty(unique_potentials)
        max_potential_val = max(unique_potentials(unique_potentials > 0 & ~isnan(unique_potentials)));
    end
    if isempty(max_potential_val) || max_potential_val == 0, max_potential_val = 1; end
    % Colormap: gray for -1 (unused), black for -2 (unusable), then distinct colors
    % 颜色图：-1（未使用）为灰色，-2（不可用）为黑色，然后是不同的颜色。
    % Using +3 because we map -2 to color 1, -1 to color 2, 0 to color 3 (if 0 is a potential)
    % 使用 +3 是因为我们将 -2 映射到颜色 1，-1 映射到颜色 2，0 映射到颜色 3（如果 0 是电位）。
    % then 1 to color 4, etc.
    % 然后 1 映射到颜色 4，依此类推。
    
    % Create a discrete colormap
    % 创建一个离散颜色图。
    num_colors_for_potentials = max_potential_val; % Number of distinct positive potentials
    % 唯一正电位的数量。
    cmap = [0 0 0;         % Color for -2 (black)
            0.5 0.5 0.5;   % Color for -1 (gray)
            lines(num_colors_for_potentials)]; % Colors for potentials 1 through max_potential_val
    % 定义颜色图：黑色（-2），灰色（-1），然后是线形颜色图（用于电位）。
    figure('Name', 'Layout Visualization', 'NumberTitle', 'off', 'WindowState', 'maximized');
    % 创建一个新的图窗，设置名称和最大化窗口。
    
    % Determine subplot layout (e.g., 3 rows or 4 rows)
    % 确定子图布局（例如，3 行或 4 行）。
    plot_cols = ceil(sqrt(num_layers));
    plot_rows = ceil(num_layers / plot_cols);
    for i = 1:num_layers
    % 遍历所有层。
        subplot(plot_rows, plot_cols, i);
        % 创建子图。
        layer_name = layer_info.names{i};
        current_layer_data = layout.(layer_name);
        % 获取当前层的数据。
        
        % Map data values to colormap indices
        % 将数据值映射到颜色图索引。
        display_data = zeros(size(current_layer_data));
        display_data(current_layer_data == -2) = 1; % Index 1 in cmap
        display_data(current_layer_data == -1) = 2; % Index 2 in cmap
        
        positive_potentials = current_layer_data > 0;
        display_data(positive_potentials) = current_layer_data(positive_potentials) + 2; % Indices 3 onwards
        % 将 -2 映射到 1，-1 映射到 2，正电位映射到其值 + 2。
        
        imagesc(display_data);
        % 显示图像。
        colormap(cmap);
        % 应用颜色图。
        clim([1, size(cmap,1)]); % Set color limits based on the number of colors in cmap
        % 设置颜色限制。
        
        title(layer_name, 'Interpreter', 'none');
        % 设置子图标题。
        axis equal tight;
        % 设置坐标轴等比例且紧密。
        set(gca, 'XTick', [], 'YTick', []);
        % 移除 X 和 Y 轴刻度。
        % Add text annotations for potentials (can be slow for large grids)
        % 为电位添加文本注释（对于大网格可能很慢）。
        if layer_info.track_height <= 20 && layer_info.L <= 20 % Only for small grids
        % 仅在小网格时添加文本注释。
            for r = 1:size(current_layer_data,1)
                for c_ = 1:size(current_layer_data,2)
                    val = current_layer_data(r,c_);
                    if val ~= -1 && val ~= -2
                        text(c_, r, num2str(val), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 8, 'Color', 'w');
                        % 在单元格上添加电位值文本。
                    end
                end
            end
        end
    end
    
    % Add a colorbar legend manually if possible (complex for discrete mapped colors)
    % 或者，如果空间允许，列出电位及其颜色。
    % Or, list potentials and their colors if space.
    % 如果可能，手动添加颜色条图例（对于离散映射颜色很复杂）。
end
