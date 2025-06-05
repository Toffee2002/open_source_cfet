% main_script.m
% This script will orchestrate the entire process.


function code3_cfet_router_v2_1(input_data_cell,design_rule)
% --- Parameters ---
track_height = 5; % Default or user-settable

% --- Input Data (Example from prompt) ---
% Ensure input_data is a numeric matrix. Handle NaNs appropriately for kruskal.
% input_data_cell = {
%     [2, 1, 5, 6, 2, 9, 8, 5, 2, 8, 10];
%     [2, 1, 3, NaN, 2, 3, 7, 6, 2, 7, 10];
%     [5, 1, 11, 6, 4, 9, 13, 5, 8, NaN, 14, 8, 4];
%     [7, 3, 12, 6, 4, 1, 3, NaN, 10, 7, 14]
% };

% Determine max L and pad shorter rows with NaN for consistent L
max_L_initial = 0;
for i = 1:numel(input_data_cell)
    if length(input_data_cell{i}) > max_L_initial
        max_L_initial = length(input_data_cell{i});
    end
end

input_data_matrix = nan(4, max_L_initial);
for i = 1:4
    row_len = length(input_data_cell{i});
    input_data_matrix(i, 1:row_len) = input_data_cell{i};
end
L = max_L_initial;


% --- 1. Initialize Layout ---
fprintf('1. Initializing Layout...\n');
layer_info = define_layer_info(track_height, L);
layout = initialize_layout(input_data_matrix, track_height, L, layer_info);
fprintf('Layout Initialized.\n\n');

% --- Design Rule for Kruskal MST (example values) ---
% design_rule = struct('a',1, 'b',2, 'c',3, 'd',100, ...
%                      'e1',1,'e2',1, 'f1',2,'f2',1, ...
%                      'g1',3,'g2',1, 'h1',4,'h2',1, 'i',0.1);


% --- 2. Identify Nets and Target Connections using User's Kruskal functions ---
fprintf('2. Identifying Nets and Target Connections...\n');
% The 'code2_calculate_total_weight' function internally uses 'kruskal_mst'.
% We need to modify/wrap 'kruskal_mst' to return the actual MST edges.

% Modified kruskal_mst to return edges
[nets_to_route, all_initial_pins] = get_routing_tasks_from_input(input_data_matrix, design_rule, track_height, layer_info);
fprintf('%d unique potentials with multiple pins identified.\n', numel(nets_to_route));
for i=1:numel(nets_to_route)
    fprintf('  Potential %d needs %d segments to be routed.\n', nets_to_route(i).potential, size(nets_to_route(i).segments,1));
end
fprintf('Nets Identified.\n\n');


% --- 3. Iterative Routing ---
fprintf('3. Starting Iterative Routing...\n');
max_routing_iterations = 5; % Main loop for rip-up and reroute
max_attempts_per_segment = 3; % Attempts for a single segment before trying rip-up

% Add status and attempt counts to nets_to_route segments
for i = 1:numel(nets_to_route)
    num_segments = size(nets_to_route(i).segments, 1);
    nets_to_route(i).segment_status = cell(num_segments, 1); % 'pending', 'routed', 'failed'
    nets_to_route(i).segment_paths = cell(num_segments, 1);
    nets_to_route(i).segment_attempts = zeros(num_segments, 1);
    for j=1:num_segments
        nets_to_route(i).segment_status{j} = 'pending';
    end
end

for iter = 1:max_routing_iterations
    fprintf('--- Routing Iteration %d ---\n', iter);
    segments_routed_this_iteration = 0;
    has_pending_segments = false;

    for net_idx = 1:numel(nets_to_route)
        potential = nets_to_route(net_idx).potential;
        for seg_idx = 1:size(nets_to_route(net_idx).segments, 1)
            if strcmp(nets_to_route(net_idx).segment_status{seg_idx}, 'routed')
                continue; % Already routed
            end
            has_pending_segments = true;

            if nets_to_route(net_idx).segment_attempts(seg_idx) >= max_attempts_per_segment
                continue; % Tried too many times for this segment in its current state
            end
            nets_to_route(net_idx).segment_attempts(seg_idx) = nets_to_route(net_idx).segment_attempts(seg_idx) + 1;

            segment = nets_to_route(net_idx).segments(seg_idx,:); % [p1_idx, p2_idx] referring to all_initial_pins
            pin1_coord_3d = all_initial_pins(segment(1)).coord_3d;
            pin2_coord_3d = all_initial_pins(segment(2)).coord_3d;

            fprintf('Attempting to route Potential %d: (%d,%d,%s) to (%d,%d,%s)\n', potential, ...
                pin1_coord_3d(1), pin1_coord_3d(2), layer_info.names{pin1_coord_3d(3)}, ...
                pin2_coord_3d(1), pin2_coord_3d(2), layer_info.names{pin2_coord_3d(3)});

            % --- A* Pathfinding ---
            % The A* should find path from any point of current potential connected to pin1_coord_3d
            % to pin2_coord_3d, or vice-versa.
            % For simplicity here, we assume points are distinct and try to connect them directly.
            % A more robust A* takes the entire existing net component as a possible start.

            [path_coords, ~] = pathfinder_astar(layout, pin1_coord_3d, pin2_coord_3d, potential, layer_info, L, track_height);

            if ~isempty(path_coords)
                % Apply path
                [layout, success_apply] = apply_path_to_layout(layout, path_coords, potential, layer_info);
                if success_apply
                    nets_to_route(net_idx).segment_status{seg_idx} = 'routed';
                    nets_to_route(net_idx).segment_paths{seg_idx} = path_coords;
                    segments_routed_this_iteration = segments_routed_this_iteration + 1;
                    fprintf('  SUCCESSFULLY Routed.\n');
                else
                    nets_to_route(net_idx).segment_status{seg_idx} = 'failed_apply_conflict';
                     fprintf('  FAILED to apply path (conflict during apply).\n');
                end
            else
                nets_to_route(net_idx).segment_status{seg_idx} = 'failed_no_path';
                fprintf('  FAILED to find path.\n');

                % --- Rip-up and Reroute (Basic Strategy) ---
                if iter < max_routing_iterations % Don't rip-up on the last iteration usually
                    % Identify a blocking net (this is complex; A* should ideally give hints)
                    % For simplicity: if path for P_A fails, and it crosses P_B, try ripping P_B.
                    % This requires detailed conflict analysis from A* or a probing mechanism.
                    % As a placeholder: if a segment fails, subsequent iterations might resolve it if other segments move.
                    % A more active rip-up would be:
                    % [blocker_net_idx, blocker_seg_idx] = find_blocker_for_segment(layout, pin1_coord_3d, pin2_coord_3d, layer_info, nets_to_route);
                    % if ~isempty(blocker_net_idx)
                    %    fprintf('   Attempting to rip-up blocking segment from potential %d\n', nets_to_route(blocker_net_idx).potential);
                    %    layout = remove_path_from_layout(layout, nets_to_route(blocker_net_idx).segment_paths{blocker_seg_idx}, layer_info);
                    %    nets_to_route(blocker_net_idx).segment_status{blocker_seg_idx} = 'pending_ripped';
                    %    nets_to_route(blocker_net_idx).segment_attempts(blocker_seg_idx) = 0; % Reset attempts for ripped segment
                    % end
                end
            end
        end
    end
    if ~has_pending_segments
         fprintf('All segments appear routed or maxed out attempts after iteration %d.\n', iter);
         break;
    end
    if segments_routed_this_iteration == 0 && iter > 1 % Check if any progress was made
        fprintf('No segments were successfully routed in iteration %d. May indicate persistent blockage.\n', iter);
        % Could stop early, or continue if rip-up is more aggressive.
    end
end
fprintf('Routing Phase Completed.\n\n');

% --- 4. Final Connectivity Check ---
fprintf('4. Performing Final Connectivity Checks...\n');
all_potentials_in_layout = [];
for i = 1:numel(all_initial_pins) % Get all relevant potentials
    all_potentials_in_layout = [all_potentials_in_layout, all_initial_pins(i).potential];
end
unique_potentials = unique(all_potentials_in_layout);

for i = 1:numel(unique_potentials)
    p_val = unique_potentials(i);
    if isnan(p_val), continue; end

    num_initial_pins_for_potential = 0;
    for k=1:numel(all_initial_pins)
        if all_initial_pins(k).potential == p_val
            num_initial_pins_for_potential = num_initial_pins_for_potential + 1;
        end
    end

    if num_initial_pins_for_potential <= 1
        fprintf('Potential %d: has %d pin(s), no complex connectivity check needed.\n', p_val, num_initial_pins_for_potential);
        continue;
    end

    [is_connected, component_size, total_size] = check_potential_connectivity_bfs(layout, p_val, layer_info);
    if is_connected
        fprintf('Potential %d: IS CONNECTED. (Component size: %d, Total cells: %d)\n', p_val, component_size, total_size);
    else
        fprintf('Potential %d: IS NOT FULLY CONNECTED. (Component size: %d, Total cells: %d)\n', p_val, component_size, total_size);
        % Here, one could trigger more advanced global or local rerouting diagnostics/attempts.
    end
end
fprintf('Connectivity Checks Completed.\n\n');

% --- 5. Visualization ---
fprintf('5. Visualizing Layout...\n');
visualize_layout(layout, layer_info, unique_potentials);
fprintf('Visualization Generated.\n');
end



% --- Helper function: Define Layer Information ---
function layer_info = define_layer_info(track_height, L)
    layer_names = {
        'FM2', 'FV_M1_M2', 'FM1', 'FV_M0_M1', 'FM0', 'FV_MD_M0', 'FMD', ...
        'V_FMD_BMD', ...
        'BMD', 'BV_MD_M0', 'BM0', 'BV_M0_M1', 'BM1', 'BV_M1_M2', 'BM2'
    };
    % H: Horizontal metal, V: Vertical metal, VIA: Via
    layer_types = {
        'H', 'VIA', 'V', 'VIA', 'H', 'VIA', 'V', ... % Top stack (F)
        'VIA', ...                                  % Connecting FMD & BMD
        'V', 'VIA', 'H', 'VIA', 'V', 'VIA', 'H'     % Bottom stack (B)
    };
    layer_info.names = layer_names;
    layer_info.types = layer_types;
    layer_info.track_height = track_height;
    layer_info.L = L;
    layer_info.name_to_idx = containers.Map(layer_names, 1:numel(layer_names));
    layer_info.idx_to_name = containers.Map(1:numel(layer_names), layer_names);
end

% --- Helper function: Initialize Layout ---
function layout = initialize_layout(input_data, track_height, L, layer_info)
    layout = struct();
    for i = 1:numel(layer_info.names)
        layout.(layer_info.names{i}) = ones(track_height, L) * -1; % -1 for unused
    end

    % Fill initial data
    % input_data(1,:) -> layout.FMD (row 2)
    % input_data(2,:) -> layout.BMD (row 2)
    % input_data(3,:) -> layout.FMD (row track_height-1)
    % input_data(4,:) -> layout.BMD (row track_height-1)
    if size(input_data,2) == L
        layout.FMD(2, :) = input_data(1, :);
        layout.BMD(2, :) = input_data(2, :);
        layout.FMD(track_height - 1, :) = input_data(3, :);
        layout.BMD(track_height - 1, :) = input_data(4, :);
    else
        warning('Input data L does not match layout L. Skipping initial data placement in FMD/BMD rows from input_data.');
    end


    % layout.V_FMD_BMD (rows 2 to track_height-1) are -2 (unusable for routing through these cells initially)
    % However, the problem statement says "V_FMD_BMD连接FMD和BMD".
    % This implies these cells *are* the vias. They are not "unusable space" but rather the via layer itself.
    % The rule is: if FMD(r,c) and BMD(r,c) need to connect, then V_FMD_BMD(r,c) must also be part of that net.
    % The constraint is on *which rows* of V_FMD_BMD can be used.
    % "layout.V_FMD_BMD的第二行到倒数第二行填-2" - this means these specific via locations CANNOT be used.
    % This is a strong constraint. Let's follow it.
    if track_height >= 3
         layout.V_FMD_BMD(2:(track_height-1), :) = -2;
    end
end

% --- Helper function to get 3D coordinates of initial pins and MST-based segments ---
function [nets_to_route, all_pins_info] = get_routing_tasks_from_input(input_data_matrix, design_rule, track_height, layer_info)
    nets_to_route = struct('potential', {}, 'segments', {}); % segments are pairs of indices into all_pins_info
    all_pins_info = struct('id', {}, 'potential', {}, 'original_loc', {}, 'coord_3d', {}); % original_loc = [row_in_input, L]
    pin_counter = 0;

    % 1. Collect all unique pin locations from input_data_matrix
    value_locations_map = containers.Map('KeyType', 'double', 'ValueType', 'any');
    for r = 1:4 % Iterate through the 4 rows of input_data_matrix
        for c = 1:size(input_data_matrix, 2)
            val = input_data_matrix(r, c);
            if ~isnan(val)
                pin_counter = pin_counter + 1;
                all_pins_info(pin_counter).id = pin_counter;
                all_pins_info(pin_counter).potential = val;
                all_pins_info(pin_counter).original_loc = [r, c];
                all_pins_info(pin_counter).coord_3d = map_2d_pin_to_3d_coord([r,c], track_height, layer_info);

                if ~isKey(value_locations_map, val)
                    value_locations_map(val) = [];
                end
                % Store the ID of the pin in all_pins_info
                value_locations_map(val) = [value_locations_map(val), pin_counter];
            end
        end
    end

    % 2. For each potential, run Kruskal to find MST edges
    unique_potentials = keys(value_locations_map);
    net_idx_counter = 0;

    for i = 1:length(unique_potentials)
        potential_val = unique_potentials{i};
        pin_ids_for_potential = value_locations_map(potential_val);

        if length(pin_ids_for_potential) <= 1
            continue; % No connections needed for single pins
        end

        % Prepare locations for kruskal_mst_get_edges_wrapper
        % The original kruskal function takes locations as [row_in_input, L_col]
        kruskal_locations = zeros(length(pin_ids_for_potential), 2);
        for k = 1:length(pin_ids_for_potential)
            kruskal_locations(k,:) = all_pins_info(pin_ids_for_potential(k)).original_loc;
        end

        % This wrapper should call your kruskal_mst and extract edges
        % Each edge is [idx1, idx2] referring to rows in kruskal_locations
        mst_edges_indices = kruskal_mst_get_edges_wrapper(kruskal_locations, design_rule);

        if ~isempty(mst_edges_indices)
            net_idx_counter = net_idx_counter + 1;
            nets_to_route(net_idx_counter).potential = potential_val;
            % Convert MST edge indices (relative to kruskal_locations) to global pin IDs
            segments_for_this_net = zeros(size(mst_edges_indices,1), 2);
            for k=1:size(mst_edges_indices,1)
                segments_for_this_net(k,1) = pin_ids_for_potential(mst_edges_indices(k,1));
                segments_for_this_net(k,2) = pin_ids_for_potential(mst_edges_indices(k,2));
            end
            nets_to_route(net_idx_counter).segments = segments_for_this_net;
        end
    end
end


function coord_3d = map_2d_pin_to_3d_coord(pin_2d, track_height, layer_info)
    % pin_2d is [input_row_idx, L_col]
    row_in_input = pin_2d(1);
    col_in_layer = pin_2d(2);
    r_in_layer = -1; layer_name_str = '';

    if row_in_input == 1
        r_in_layer = 2; layer_name_str = 'FMD';
    elseif row_in_input == 2
        r_in_layer = 2; layer_name_str = 'BMD';
    elseif row_in_input == 3
        r_in_layer = track_height - 1; layer_name_str = 'FMD';
    elseif row_in_input == 4
        r_in_layer = track_height - 1; layer_name_str = 'BMD';
    else
        error('Invalid input_row_idx for pin mapping: %d', row_in_input);
    end
    layer_idx = layer_info.name_to_idx(layer_name_str);
    coord_3d = [r_in_layer, col_in_layer, layer_idx];
end


% --- KRUSKAL MST related functions (User Provided, with a wrapper) ---
% Wrapper for kruskal_mst to return edges
function mst_edges = kruskal_mst_get_edges_wrapper(locations, design_rule)
    num_nodes = size(locations, 1);
    mst_edges = [];
    if num_nodes <= 1
        return;
    end

    edges_data = []; % Store as [u, v, cost, rule_applied_char]
    for i = 1:num_nodes
        for j = i+1:num_nodes
            [cost, rule] = code2_get_pair_cost_with_rule(locations(i,:), locations(j,:), design_rule);
            if isfinite(cost) && cost >= 0
                edges_data = [edges_data; i, j, cost, rule(1)]; % rule is char
            end
        end
    end

    if isempty(edges_data)
        warning('Kruskal: No valid edges found for locations provided.');
        return;
    end

    edges_data = sortrows(edges_data, 3); % Sort by cost

    parent = 1:num_nodes;
    rank = zeros(1, num_nodes);
    num_edges_in_mst = 0;

    for k = 1:size(edges_data, 1)
        u = edges_data(k, 1);
        v = edges_data(k, 2);
        % cost_val = edges_data(k, 3); % not directly used for edge list

        root_u = find_root_kruskal(parent, u);
        root_v = find_root_kruskal(parent, v);

        if root_u ~= root_v
            mst_edges = [mst_edges; u, v]; % Add edge (indices relative to 'locations')
            num_edges_in_mst = num_edges_in_mst + 1;
            % Union sets
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
            end
        end
    end
     % Check if MST actually spans all nodes (if locations formed a single component)
    if num_nodes > 1 && num_edges_in_mst < num_nodes - 1
        % This can happen if the original points form disconnected groups
        % based on the cost function (e.g. infinite cost between some pairs)
        % The returned 'mst_edges' would be for the largest possible forest.
        % For routing, we might want to connect all points of the same potential regardless of this initial MST cost.
        % However, sticking to the prompt, the MST edges define connections.
        % warning('Kruskal: MST for %d nodes only has %d edges. Points might be partitioned by cost function.', num_nodes, num_edges_in_mst);
    end
end

function root = find_root_kruskal(parent, x) % Kruskal's Union-Find helper
    if parent(x) == x
        root = x;
    else
        parent(x) = find_root_kruskal(parent, parent(x)); % Path compression
        root = parent(x);
    end
    % Non-recursive version for deep stacks if needed:
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
% (Copied directly from prompt)
function [cost, rule_applied] = code2_get_pair_cost_with_rule(loc1, loc2, design_rule)
    row1 = loc1(1); L1 = loc1(2);
    row2 = loc2(1); L2 = loc2(2);
    cost = inf; % Default cost if no rule matches
    rule_applied = '';
    if L1 == L2 % Same L
        if row1 == row2 % This case should ideally not happen if loc1 and loc2 are distinct pins
            cost = design_rule.a; rule_applied = 'a'; % Cost of a point with itself? Or very close.
        elseif (ismember(row1, [1,2]) && ismember(row2, [1,2])) % Both in FMD/BMD top-ish rows
            cost = design_rule.b; rule_applied = 'b';
        elseif (ismember(row1, [3,4]) && ismember(row2, [3,4])) % Both in FMD/BMD bottom-ish rows
            cost = design_rule.b; rule_applied = 'b';
        elseif isequal(sort([row1, row2]), [1, 3]) % FMD top to FMD bottom (or vice versa)
            cost = design_rule.c; rule_applied = 'c';
        elseif isequal(sort([row1, row2]), [2, 4]) % BMD top to BMD bottom (or vice versa)
            cost = design_rule.c; rule_applied = 'c';
        else % e.g. [1,2] with [3,4] or [1,4] or [2,3]
            cost = design_rule.d; rule_applied = 'd';
        end
    else % Different L
        dL = abs(L1 - L2);
        rpair = sort([row1, row2]);
        if row1 == row2 % Same row, different L
            cost = design_rule.e1 + design_rule.e2 * dL; rule_applied = 'e';
        elseif isequal(rpair, [1, 2]) % FMD row with BMD row (same side, top)
            cost = design_rule.f1 + design_rule.f2 * dL; rule_applied = 'f';
        elseif isequal(rpair, [3, 4]) % FMD row with BMD row (same side, bottom)
            cost = design_rule.f1 + design_rule.f2 * dL; rule_applied = 'f';
        elseif isequal(rpair, [1, 3]) % FMD top with FMD bottom (different L)
            cost = design_rule.g1 + design_rule.g2 * dL; rule_applied = 'g';
        elseif isequal(rpair, [2, 4]) % BMD top with BMD bottom (different L)
            cost = design_rule.g1 + design_rule.g2 * dL; rule_applied = 'g';
        elseif isequal(rpair, [2, 3]) % BMD top with FMD bottom (diagonal across F/B)
            cost = design_rule.h1 + design_rule.h2 * dL; rule_applied = 'h';
        elseif isequal(rpair, [1, 4]) % FMD top with BMD bottom (diagonal across F/B)
            cost = design_rule.h1 + design_rule.h2 * dL; rule_applied = 'h';
        end
    end
    % if isinf(cost)
    %     warning('Infinite cost between loc (%d,%d) and (%d,%d)', row1, L1, row2, L2);
    % end
end


% --- A* Pathfinder ---
function [path_coords, cost] = pathfinder_astar(layout, start_coord_3d, end_coord_3d, potential, layer_info, L_max, H_max)
    % start_coord_3d, end_coord_3d: [r, c, layer_idx]
    % potential: the numeric value to route
    % layer_info: struct with names, types, etc.
    % L_max, H_max: dimensions of layers

    path_coords = [];
    cost = inf;

    start_node.g = 0;
    start_node.h = heuristic(start_coord_3d, end_coord_3d, layer_info);
    start_node.f = start_node.g + start_node.h;
    start_node.coord = start_coord_3d;
    start_node.parent_coord = []; % No parent for start

    pq = PriorityQueue(); % Needs a simple priority queue implementation
    pq.insert(start_node, start_node.f);

    closed_set = containers.Map('KeyType','char','ValueType','any'); % Store node coords (as string) that have been processed

    while ~pq.isEmpty()
        current_node_pq = pq.pop(); % Gets the element, not the priority
        current_coord_str = mat2str(current_node_pq.coord);

        if isKey(closed_set, current_coord_str)
            continue;
        end
        closed_set(current_coord_str) = current_node_pq;


        if all(current_node_pq.coord == end_coord_3d) % Goal reached
            path_coords = reconstruct_path(closed_set, current_node_pq.coord);
            cost = current_node_pq.f;
            return;
        end

        neighbors = get_valid_neighbors(layout, current_node_pq.coord, potential, layer_info, L_max, H_max);

        for i = 1:size(neighbors, 1)
            neighbor_coord = neighbors(i,:);
            neighbor_coord_str = mat2str(neighbor_coord);

            if isKey(closed_set, neighbor_coord_str) % Already processed this neighbor fully
                continue;
            end
            
            % Check if neighbor is blocked by another potential or -2
            % This check should also be inside get_valid_neighbors, but double check
            n_r = neighbor_coord(1); n_c = neighbor_coord(2); n_l_idx = neighbor_coord(3);
            layer_name = layer_info.idx_to_name(n_l_idx);
            if layout.(layer_name)(n_r, n_c) ~= -1 && layout.(layer_name)(n_r, n_c) ~= potential
                continue; % Blocked by other potential
            end
            if layout.(layer_name)(n_r, n_c) == -2
                 continue; % Blocked by -2
            end


            g_score = current_node_pq.g + 1; % Simple cost: 1 per step (vias could be more)
                                           % Vias often have higher cost, adjust '1' based on neighbor type
            
            % Check if neighbor is already in PQ with a higher g_score
            % (This part makes it a bit like Dijkstra if not careful with PQ updates,
            % but for A* it's about finding it not yet in closed_set or with a better path)

            existing_in_pq = pq.find(neighbor_coord); % Needs a find method in PQ by coord
            
            if isempty(existing_in_pq) || g_score < existing_in_pq.g
                neighbor_node.g = g_score;
                neighbor_node.h = heuristic(neighbor_coord, end_coord_3d, layer_info);
                neighbor_node.f = neighbor_node.g + neighbor_node.h;
                neighbor_node.coord = neighbor_coord;
                neighbor_node.parent_coord = current_node_pq.coord;
                
                if ~isempty(existing_in_pq)
                    pq.remove(existing_in_pq); % If PQ supports removal/update
                end
                pq.insert(neighbor_node, neighbor_node.f);
            end
        end
    end
end

function h_val = heuristic(coord1, coord2, layer_info)
    % Manhattan distance, slightly weighted by layer difference
    dr = abs(coord1(1) - coord2(1));
    dc = abs(coord1(2) - coord2(2));
    dl = abs(coord1(3) - coord2(3)); % Difference in layer indices
    
    % Basic heuristic: sum of differences.
    % More advanced: consider number of vias needed based on layer types.
    % If layers are far apart and of same type (e.g. H to H), vias are necessary.
    h_val = dr + dc + dl * 2; % Penalize layer changes a bit more
end

function neighbors = get_valid_neighbors(layout, current_coord, potential, layer_info, L_max, H_max)
    % current_coord: [r, c, layer_idx]
    r = current_coord(1); c = current_coord(2); l_idx = current_coord(3);
    current_layer_name = layer_info.idx_to_name(l_idx);
    current_layer_type = layer_info.types{l_idx};
    neighbors = zeros(0,3); % Initialize as empty

    % --- Intra-layer connections ---
    if strcmp(current_layer_type, 'H') % Horizontal layers: FM0, BM0, FM2, BM2
        moves = [0 -1; 0 1]; % Left, Right
    elseif strcmp(current_layer_type, 'V') % Vertical layers: FMD, BMD, FM1, BM1
        moves = [-1 0; 1 0]; % Up, Down
    else % VIA layer - no intra-layer routing movement in this model
        moves = [];
    end

    for i=1:size(moves,1)
        nr = r + moves(i,1);
        nc = c + moves(i,2);
        if nr >= 1 && nr <= H_max && nc >= 1 && nc <= L_max
            % Cell must be -1 (empty) or same potential (part of the net already)
            if layout.(current_layer_name)(nr,nc) == -1 || layout.(current_layer_name)(nr,nc) == potential
                neighbors(end+1,:) = [nr, nc, l_idx];
            end
        end
    end

    % --- Inter-layer connections (Vias) ---
    % 1. From a Metal layer TO an adjacent Metal layer THROUGH a VIA layer
    if strcmp(current_layer_type, 'H') || strcmp(current_layer_type, 'V')
        [via_names_up, metal_names_up, via_names_down, metal_names_down] = get_adjacent_vias_metals(l_idx, layer_info);

        for i=1:numel(via_names_up) % Connections "upwards" or to next layer in sequence
            via_l_idx = layer_info.name_to_idx(via_names_up{i});
            adj_metal_l_idx = layer_info.name_to_idx(metal_names_up{i});
            % Check if via cell and target metal cell are usable
            if (layout.(via_names_up{i})(r,c) == -1 || layout.(via_names_up{i})(r,c) == potential) && ...
               (layout.(metal_names_up{i})(r,c) == -1 || layout.(metal_names_up{i})(r,c) == potential)
                % The "neighbor" is the cell in the adjacent metal layer. The path implies using the via.
                neighbors(end+1,:) = [r, c, adj_metal_l_idx];
            end
        end
        for i=1:numel(via_names_down) % Connections "downwards"
            via_l_idx = layer_info.name_to_idx(via_names_down{i});
            adj_metal_l_idx = layer_info.name_to_idx(metal_names_down{i});
             if (layout.(via_names_down{i})(r,c) == -1 || layout.(via_names_down{i})(r,c) == potential) && ...
               (layout.(metal_names_down{i})(r,c) == -1 || layout.(metal_names_down{i})(r,c) == potential)
                neighbors(end+1,:) = [r, c, adj_metal_l_idx];
            end
        end
    end
    
    % V_FMD_BMD special handling: It connects FMD and BMD at the same (r,c)
    if strcmp(current_layer_name, 'FMD')
        via_name = 'V_FMD_BMD'; adj_metal_name = 'BMD';
        adj_metal_l_idx = layer_info.name_to_idx(adj_metal_name);
        % Check if the V_FMD_BMD cell itself is usable (-1 or current potential)
        % The problem states V_FMD_BMD(2:H-1,:) are -2. So this via is only usable at row 1 and H.
        if (layout.(via_name)(r,c) == -1 || layout.(via_name)(r,c) == potential) && ...
           (layout.(adj_metal_name)(r,c) == -1 || layout.(adj_metal_name)(r,c) == potential)
             neighbors(end+1,:) = [r, c, adj_metal_l_idx];
        end
    elseif strcmp(current_layer_name, 'BMD')
        via_name = 'V_FMD_BMD'; adj_metal_name = 'FMD';
        adj_metal_l_idx = layer_info.name_to_idx(adj_metal_name);
         if (layout.(via_name)(r,c) == -1 || layout.(via_name)(r,c) == potential) && ...
           (layout.(adj_metal_name)(r,c) == -1 || layout.(adj_metal_name)(r,c) == potential)
             neighbors(end+1,:) = [r, c, adj_metal_l_idx];
        end
    end
end


function [via_names_up, metal_names_up, via_names_down, metal_names_down] = get_adjacent_vias_metals(current_metal_layer_idx, layer_info)
    % For a given metal layer index, find directly connectable vias and the metals on their other side.
    % "Up" usually means towards FM2 or BM2 (higher index in typical layout stack order if sorted).
    % "Down" usually means towards FMD or BMD.
    % This depends on the exact ordering in layer_info.names. Let's use sequence.
    via_names_up = {}; metal_names_up = {};
    via_names_down = {}; metal_names_down = {};
    current_metal_name = layer_info.idx_to_name(current_metal_layer_idx);

    % Check layer before current_metal_layer_idx (potential via "down")
    if current_metal_layer_idx > 1
        prev_layer_idx = current_metal_layer_idx - 1;
        prev_layer_name = layer_info.idx_to_name(prev_layer_idx);
        if strcmp(layer_info.types{prev_layer_idx}, 'VIA') % If it's a VIA
            % Check if this VIA connects to current_metal_name
            [m1, m2] = get_metals_for_via(prev_layer_name, layer_info);
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
    if current_metal_layer_idx < numel(layer_info.names)
        next_layer_idx = current_metal_layer_idx + 1;
        next_layer_name = layer_info.idx_to_name(next_layer_idx);
        if strcmp(layer_info.types{next_layer_idx}, 'VIA') % If it's a VIA
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
    % Given a VIA layer name, find the two metal layers it connects.
    metal1_name = ''; metal2_name = '';
    parts = strsplit(via_layer_name, '_'); % e.g., FV_M1_M2 or V_FMD_BMD

    if numel(parts) < 3
        % warning('Via name %s does not follow expected format.', via_layer_name);
        return;
    end
    
    prefix = parts{1};
    p2 = parts{2};
    p3 = parts{3};

    if strcmp(prefix, 'FV') % e.g., FV_M1_M2 connects FM1 and FM2
        metal1_name = ['F' p2];
        metal2_name = ['F' p3];
    elseif strcmp(prefix, 'BV') % e.g., BV_M0_M1 connects BM0 and BM1
        metal1_name = ['B' p2];
        metal2_name = ['B' p3];
    elseif strcmp(prefix, 'V') && strcmp(p2,'FMD') && strcmp(p3,'BMD') % V_FMD_BMD
        metal1_name = 'FMD';
        metal2_name = 'BMD';
    else
        % warning('Unknown via naming convention for %s', via_layer_name);
    end
    
    % Validate that these metal names exist in layer_info
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
    path = current_coord;
    current_coord_str = mat2str(current_coord);
    while ~isempty(closed_set(current_coord_str).parent_coord)
        parent_c = closed_set(current_coord_str).parent_coord;
        path = [parent_c; path];
        current_coord_str = mat2str(parent_c);
        if ~isKey(closed_set, current_coord_str) % Should not happen if path reconstruction is correct
            error('Error in path reconstruction: parent not in closed set.');
        end
    end
end

% --- Apply Path to Layout ---
function [layout_out, success] = apply_path_to_layout(layout_in, path_coords, potential, layer_info)
    layout_out = layout_in;
    success = true;
    % path_coords is N x 3, where each row is [r, c, layer_idx]
    
    % First, check for conflicts along the path before applying
    for i = 1:size(path_coords,1)
        pt = path_coords(i,:);
        r = pt(1); c = pt(2); l_idx = pt(3);
        layer_name = layer_info.idx_to_name(l_idx);
        if layout_out.(layer_name)(r,c) ~= -1 && layout_out.(layer_name)(r,c) ~= potential
            success = false; % Conflict with another potential
            % disp(['Conflict applying path at (', num2str(r), ',', num2str(c), ') in layer ', layer_name, '. Expected -1 or ', num2str(potential), ', found ', num2str(layout_out.(layer_name)(r,c))]);
            return; % Do not apply if conflict
        end
    end

    % Apply path if no conflicts found in the check
    for i = 1:size(path_coords,1)
        pt = path_coords(i,:);
        r = pt(1); c = pt(2); l_idx = pt(3);
        layer_name = layer_info.idx_to_name(l_idx);
        layout_out.(layer_name)(r,c) = potential;
        
        % If this point implies a via was used, mark the via layer too.
        % The path_coords should ideally contain explicit via cells if they are separate entities.
        % Our current get_valid_neighbors implies via usage by jumping layers.
        % We need to mark the via cell between path_coords(i-1) and path_coords(i) if it's a layer jump.
        if i > 1
            prev_pt = path_coords(i-1,:);
            pr = prev_pt(1); pc = prev_pt(2); pl_idx = prev_pt(3);
            % If current point (r,c,l_idx) and previous point (pr,pc,pl_idx)
            % are at same (r,c) but different l_idx, a via was traversed.
            if r == pr && c == pc && l_idx ~= pl_idx
                % Find the via layer between l_idx and pl_idx
                via_layer_l_idx = find_via_layer_between(l_idx, pl_idx, layer_info);
                if via_layer_l_idx > 0
                    via_layer_name = layer_info.idx_to_name(via_layer_l_idx);
                    if layout_out.(via_layer_name)(r,c) ~= -1 && layout_out.(via_layer_name)(r,c) ~= potential
                         success = false; % conflict on via
                         % If conflict on via, should ideally not have happened if A* checked.
                         % For safety, revert (or don't apply from start if pre-checked)
                         % This part needs robust handling. For now, assume A* path is clear.
                         return;
                    end
                    layout_out.(via_layer_name)(r,c) = potential;
                end
            end
        end
    end
end

function via_l_idx = find_via_layer_between(metal1_l_idx, metal2_l_idx, layer_info)
    via_l_idx = 0; % Not found
    % Check layers between metal1_l_idx and metal2_l_idx in sequence
    idx_start = min(metal1_l_idx, metal2_l_idx);
    idx_end = max(metal1_l_idx, metal2_l_idx);

    if abs(metal1_l_idx - metal2_l_idx) ~= 2 && ~( (strcmp(layer_info.idx_to_name(metal1_l_idx),'FMD') && strcmp(layer_info.idx_to_name(metal2_l_idx),'BMD')) || ...
                                                 (strcmp(layer_info.idx_to_name(metal2_l_idx),'FMD') && strcmp(layer_info.idx_to_name(metal1_l_idx),'BMD')) )
        % Typically, metal layers are separated by one via layer in sequence
        % Special case for V_FMD_BMD which might not be in direct sequence
         if (strcmp(layer_info.idx_to_name(metal1_l_idx),'FMD') && strcmp(layer_info.idx_to_name(metal2_l_idx),'BMD')) || ...
            (strcmp(layer_info.idx_to_name(metal2_l_idx),'FMD') && strcmp(layer_info.idx_to_name(metal1_l_idx),'BMD'))
            via_l_idx = layer_info.name_to_idx('V_FMD_BMD');
            return;
         end
        return; % Not adjacent through a single via in sequence
    end
    
    potential_via_idx = (idx_start + idx_end) / 2;
    if round(potential_via_idx) == potential_via_idx && strcmp(layer_info.types{potential_via_idx}, 'VIA')
        % Check if this via connects these specific metals
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
    if (strcmp(layer_info.idx_to_name(metal1_l_idx),'FMD') && strcmp(layer_info.idx_to_name(metal2_l_idx),'BMD')) || ...
       (strcmp(layer_info.idx_to_name(metal2_l_idx),'FMD') && strcmp(layer_info.idx_to_name(metal1_l_idx),'BMD'))
        via_l_idx = layer_info.name_to_idx('V_FMD_BMD');
    end
end


% --- Rip-up Path from Layout (Simplified) ---
function layout_out = remove_path_from_layout(layout_in, path_coords, layer_info)
    % path_coords is N x 3 from a previously routed segment
    layout_out = layout_in;
    if isempty(path_coords)
        return;
    end
    for i = 1:size(path_coords,1)
        pt = path_coords(i,:);
        r = pt(1); c = pt(2); l_idx = pt(3);
        layer_name = layer_info.idx_to_name(l_idx);
        % Only set to -1 if it currently holds the potential of the path being removed.
        % This is tricky if multiple paths of same potential overlap.
        % For rip-up, assume we are removing this specific instance.
        layout_out.(layer_name)(r,c) = -1; % Set back to unused

        % Also clear implied vias
        if i > 1
            prev_pt = path_coords(i-1,:);
            pr = prev_pt(1); pc = prev_pt(2); pl_idx = prev_pt(3);
            if r == pr && c == pc && l_idx ~= pl_idx
                via_layer_l_idx = find_via_layer_between(l_idx, pl_idx, layer_info);
                if via_layer_l_idx > 0
                    via_layer_name = layer_info.idx_to_name(via_layer_l_idx);
                    layout_out.(via_layer_name)(r,c) = -1;
                end
            end
        end
    end
end


% --- Connectivity Check (BFS-based) ---
function [is_connected, component_count, total_potential_cells] = check_potential_connectivity_bfs(layout, potential_value, layer_info)
    H_max = layer_info.track_height;
    L_max = layer_info.L;
    
    q = java.util.LinkedList();
    visited_map = containers.Map('KeyType','char','ValueType','logical');
    
    first_point_found = false;
    start_coord = [];
    total_potential_cells = 0;

    % Find all cells with the potential_value and count them / find a starting point
    for l_idx = 1:numel(layer_info.names)
        layer_name = layer_info.idx_to_name(l_idx);
        [rows, cols] = find(layout.(layer_name) == potential_value);
        for k=1:numel(rows)
            total_potential_cells = total_potential_cells + 1;
            if ~first_point_found
                start_coord = [rows(k), cols(k), l_idx];
                q.add(start_coord);
                visited_map(mat2str(start_coord)) = true;
                first_point_found = true;
            end
        end
    end

    if total_potential_cells == 0 || total_potential_cells == 1
        is_connected = true;
        component_count = total_potential_cells;
        return;
    end
    
    if ~first_point_found % Should not happen if total_potential_cells > 0
        is_connected = false; % Or true if 0 cells
        component_count = 0;
        return;
    end

    component_count = 0;
    while ~q.isEmpty()
        current_coord = q.remove();
        component_count = component_count + 1;
        
        % Get neighbors (similar to A* but only on same potential)
        neighbors = get_connected_neighbors_for_check(layout, current_coord, potential_value, layer_info, L_max, H_max);
        
        for i=1:size(neighbors,1)
            neighbor_coord = neighbors(i,:);
            neighbor_coord_str = mat2str(neighbor_coord);
            if ~isKey(visited_map, neighbor_coord_str)
                % Check if the neighbor actually has the potential value
                % (get_connected_neighbors_for_check should ensure this)
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
end

function neighbors = get_connected_neighbors_for_check(layout, current_coord, potential, layer_info, L_max, H_max)
    % Similar to get_valid_neighbors but for connectivity check (must be same potential)
    r = current_coord(1); c = current_coord(2); l_idx = current_coord(3);
    current_layer_name = layer_info.idx_to_name(l_idx);
    current_layer_type = layer_info.types{l_idx};
    neighbors = zeros(0,3);

    % Intra-layer
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
            if layout.(via_names_down{i})(r,c) == potential && layout.(metal_names_down{i})(r,c) == potential
                neighbors(end+1,:) = [r, c, adj_metal_l_idx];
            end
        end
    end
    % V_FMD_BMD specific check
    if strcmp(current_layer_name, 'FMD') && layout.V_FMD_BMD(r,c) == potential && layout.BMD(r,c) == potential
        neighbors(end+1,:) = [r, c, layer_info.name_to_idx('BMD')];
    elseif strcmp(current_layer_name, 'BMD') && layout.V_FMD_BMD(r,c) == potential && layout.FMD(r,c) == potential
        neighbors(end+1,:) = [r, c, layer_info.name_to_idx('FMD')];
    end
end


% --- Visualization ---
function visualize_layout(layout, layer_info, unique_potentials)
    num_layers = numel(layer_info.names);
    
    % Create a colormap
    % Max potential value for colormap scaling. Add 2 for -1 and -2.
    max_potential_val = 0;
    if ~isempty(unique_potentials)
        max_potential_val = max(unique_potentials(unique_potentials > 0 & ~isnan(unique_potentials)));
    end
    if isempty(max_potential_val) || max_potential_val == 0, max_potential_val = 1; end

    % Colormap: gray for -1 (unused), black for -2 (unusable), then distinct colors
    % Using +3 because we map -2 to color 1, -1 to color 2, 0 to color 3 (if 0 is a potential)
    % then 1 to color 4, etc.
    
    % Shift data for colormapping: map -2 to 1, -1 to 2, val to val+2
    % Create a discrete colormap
    num_colors_for_potentials = max_potential_val; % Number of distinct positive potentials
    cmap = [0 0 0;         % Color for -2 (black)
            0.5 0.5 0.5;   % Color for -1 (gray)
            lines(num_colors_for_potentials)]; % Colors for potentials 1 through max_potential_val

    figure('Name', 'Layout Visualization', 'NumberTitle', 'off', 'WindowState', 'maximized');
    
    % Determine subplot layout (e.g., 3 rows or 4 rows)
    plot_cols = ceil(sqrt(num_layers));
    plot_rows = ceil(num_layers / plot_cols);

    for i = 1:num_layers
        subplot(plot_rows, plot_cols, i);
        layer_name = layer_info.names{i};
        current_layer_data = layout.(layer_name);
        
        % Map data values to colormap indices
        display_data = zeros(size(current_layer_data));
        display_data(current_layer_data == -2) = 1; % Index 1 in cmap
        display_data(current_layer_data == -1) = 2; % Index 2 in cmap
        
        positive_potentials = current_layer_data > 0;
        display_data(positive_potentials) = current_layer_data(positive_potentials) + 2; % Indices 3 onwards
        
        imagesc(display_data);
        colormap(cmap);
        clim([1, size(cmap,1)]); % Set color limits based on the number of colors in cmap
        
        title(layer_name, 'Interpreter', 'none');
        axis equal tight;
        set(gca, 'XTick', [], 'YTick', []);

        % Add text annotations for potentials (can be slow for large grids)
        if layer_info.track_height <= 20 && layer_info.L <= 20 % Only for small grids
            for r = 1:size(current_layer_data,1)
                for c_ = 1:size(current_layer_data,2)
                    val = current_layer_data(r,c_);
                    if val ~= -1 && val ~= -2
                        text(c_, r, num2str(val), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 8, 'Color', 'w');
                    end
                end
            end
        end
    end
    
    % Add a colorbar legend manually if possible (complex for discrete mapped colors)
    % Or, list potentials and their colors if space.
end
