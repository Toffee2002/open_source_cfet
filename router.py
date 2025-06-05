import csv  # 导入CSV文件处理模块
import math  # 导入数学函数模块，用于 math.pow 和 math.factorial
from z3 import Solver, Int, Bool, And, Or, Distinct, Implies, Sum, sat, Not, is_true, If  # 导入Z3约束求解器的相关函数


# 辅助类：存储晶体管数据及其原始索引和类型
class TransistorInfo:
    def __init__(self, sd1, sd2, gate, original_index, type_char):
        self.sd1 = sd1  # 晶体管的第一个源极/漏极端子
        self.sd2 = sd2  # 晶体管的第二个源极/漏极端子
        self.gate = gate  # 晶体管的栅极端子
        self.original_index = original_index  # 晶体管在原始数据中的索引
        self.type_char = type_char  # 晶体管类型：'N' 表示NMOS，'P' 表示PMOS

    def __repr__(self):
        # 定义对象的字符串表示形式，便于调试和打印
        return f"T({self.type_char}{self.original_index}: {self.sd1},{self.sd2},{self.gate})"


# 辅助函数：根据有序的晶体管（含方向）构建实际的行输出
def build_row_output(ordered_transistors_with_details, nan_placeholder="NaN"):
    row_elements = []  # 初始化行元素列表
    if not ordered_transistors_with_details:  # 如果没有晶体管，返回空列表
        return row_elements

    # 处理第一个晶体管
    first_t_data = ordered_transistors_with_details[0]  # 获取第一个晶体管数据
    # 根据方向确定有效的源极和漏极
    s_eff = first_t_data['trans_info'].sd2 if first_t_data['orientation_is_reversed'] else first_t_data[
        'trans_info'].sd1
    d_eff = first_t_data['trans_info'].sd1 if first_t_data['orientation_is_reversed'] else first_t_data[
        'trans_info'].sd2
    g_eff = first_t_data['trans_info'].gate  # 栅极不受方向影响

    row_elements.extend([s_eff, g_eff, d_eff])  # 将第一个晶体管的 源极-栅极-漏极 添加到行中
    last_sd_placed = d_eff  # 记录上一个放置的源极/漏极端子

    # 处理剩余的晶体管
    for i in range(1, len(ordered_transistors_with_details)):
        current_t_data = ordered_transistors_with_details[i]  # 获取当前晶体管数据
        # 根据方向确定当前晶体管的有效源极和漏极
        s_curr = current_t_data['trans_info'].sd2 if current_t_data['orientation_is_reversed'] else current_t_data[
            'trans_info'].sd1
        d_curr = current_t_data['trans_info'].sd1 if current_t_data['orientation_is_reversed'] else current_t_data[
            'trans_info'].sd2
        g_curr = current_t_data['trans_info'].gate

        if last_sd_placed == s_curr:  # 如果当前晶体管的源极与上一个晶体管的漏极相连
            row_elements.extend([g_curr, d_curr])  # 只需添加栅极和漏极
        else:  # 如果不相连，需要插入占位符和完整的晶体管信息
            row_elements.extend([nan_placeholder, s_curr, g_curr, d_curr])
        last_sd_placed = d_curr  # 更新上一个放置的端子

    return row_elements  # 返回构建好的行元素列表


def solve_transistor_placements(DataN_input, DataP_input, max_unique_solutions_count=25000):
    """
    主求解函数：使用Z3求解器找到晶体管布局的所有可能解
    DataN_input: NMOS晶体管数据列表
    DataP_input: PMOS晶体管数据列表
    max_unique_solutions_count: 最大唯一解数量限制
    """
    s = Solver()  # 创建Z3求解器实例
    base_configurations_from_z3 = []  # 存储Z3找到的基础配置

    # 将输入数据转换为TransistorInfo对象
    DataN_transistors = [TransistorInfo(d[0], d[1], d[2], i, 'N') for i, d in enumerate(DataN_input)]
    DataP_transistors = [TransistorInfo(d[0], d[1], d[2], i, 'P') for i, d in enumerate(DataP_input)]

    len_N = len(DataN_transistors)  # NMOS晶体管数量
    len_P = len(DataP_transistors)  # PMOS晶体管数量

    # --- 阶段1: Z3 找出 (行, 顺序) 的基础分配 ---
    # 为每个NMOS晶体管创建行和顺序变量
    n_row_vars = [Int(f"n_row_{i}") for i in range(len_N)]  # NMOS晶体管的行分配变量
    n_order_vars = [Int(f"n_order_{i}") for i in range(len_N)]  # NMOS晶体管的顺序变量
    # 为每个PMOS晶体管创建行和顺序变量
    p_row_vars = [Int(f"p_row_{i}") for i in range(len_P)]  # PMOS晶体管的行分配变量
    p_order_vars = [Int(f"p_order_{i}") for i in range(len_P)]  # PMOS晶体管的顺序变量
    # 每行中晶体管数量的变量（4行：N0, N1, P0, P1）
    num_in_row_vars = [Int(f"num_in_row_{r}") for r in range(4)]

    # DataN 的约束条件 (仅行和顺序)
    if len_N > 0:  # 如果有NMOS晶体管
        s.add(num_in_row_vars[0] >= 0, num_in_row_vars[1] >= 0)  # 行中晶体管数量非负
        s.add(num_in_row_vars[0] + num_in_row_vars[1] == len_N)  # 两行的晶体管总数等于NMOS总数
        # 计算第0行中的NMOS晶体管数量
        s.add(num_in_row_vars[0] == Sum([If(n_row_vars[i] == 0, 1, 0) for i in range(len_N)]))
        # 计算第1行中的NMOS晶体管数量
        s.add(num_in_row_vars[1] == Sum([If(n_row_vars[i] == 1, 1, 0) for i in range(len_N)]))

        for i in range(len_N):  # 为每个NMOS晶体管添加约束
            s.add(Or(n_row_vars[i] == 0, n_row_vars[i] == 1))  # 每个晶体管只能在第0行或第1行
            # 如果在第0行，顺序必须在有效范围内
            s.add(Implies(n_row_vars[i] == 0, And(n_order_vars[i] >= 0, n_order_vars[i] < num_in_row_vars[0])))
            # 如果在第1行，顺序必须在有效范围内
            s.add(Implies(n_row_vars[i] == 1, And(n_order_vars[i] >= 0, n_order_vars[i] < num_in_row_vars[1])))
            s.add(n_order_vars[i] >= 0, n_order_vars[i] < len_N)  # 顺序变量的基本约束

        # 确保同一行中的晶体管有不同的顺序
        for r_idx in range(2):  # 对于第0行和第1行
            for i in range(len_N):
                for j in range(i + 1, len_N):
                    s.add(Implies(And(n_row_vars[i] == r_idx, n_row_vars[j] == r_idx),
                                  n_order_vars[i] != n_order_vars[j]))
    else:  # 如果没有NMOS晶体管
        s.add(num_in_row_vars[0] == 0, num_in_row_vars[1] == 0)  # 相应行的晶体管数量为0

    # DataP 的约束条件 (仅行和顺序)
    if len_P > 0:  # 如果有PMOS晶体管
        s.add(num_in_row_vars[2] >= 0, num_in_row_vars[3] >= 0)  # 行中晶体管数量非负
        s.add(num_in_row_vars[2] + num_in_row_vars[3] == len_P)  # 两行的晶体管总数等于PMOS总数
        # 计算第2行中的PMOS晶体管数量
        s.add(num_in_row_vars[2] == Sum([If(p_row_vars[i] == 2, 1, 0) for i in range(len_P)]))
        # 计算第3行中的PMOS晶体管数量
        s.add(num_in_row_vars[3] == Sum([If(p_row_vars[i] == 3, 1, 0) for i in range(len_P)]))

        for i in range(len_P):  # 为每个PMOS晶体管添加约束
            s.add(Or(p_row_vars[i] == 2, p_row_vars[i] == 3))  # 每个晶体管只能在第2行或第3行
            # 如果在第2行，顺序必须在有效范围内
            s.add(Implies(p_row_vars[i] == 2, And(p_order_vars[i] >= 0, p_order_vars[i] < num_in_row_vars[2])))
            # 如果在第3行，顺序必须在有效范围内
            s.add(Implies(p_row_vars[i] == 3, And(p_order_vars[i] >= 0, p_order_vars[i] < num_in_row_vars[3])))
            s.add(p_order_vars[i] >= 0, p_order_vars[i] < len_P)  # 顺序变量的基本约束

        # 确保同一行中的晶体管有不同的顺序
        for r_offset in range(2):  # 对于第2行和第3行
            actual_row_index = r_offset + 2  # 实际行索引：2或3
            for i in range(len_P):
                for j in range(i + 1, len_P):
                    s.add(Implies(And(p_row_vars[i] == actual_row_index, p_row_vars[j] == actual_row_index),
                                  p_order_vars[i] != p_order_vars[j]))
    else:  # 如果没有PMOS晶体管
        s.add(num_in_row_vars[2] == 0, num_in_row_vars[3] == 0)  # 相应行的晶体管数量为0

    print("阶段1: Z3 正在寻找基础配置 (行和顺序分配)...")
    base_configs_found_count = 0  # 已找到的基础配置计数
    # 计算理论基础配置数: ((L_N+1)*L_N!) * ((L_P+1)*L_P!)
    # 例如 L_N=4, L_P=3 -> (5*24)*(4*6) = 120 * 24 = 2880
    # 设置一个略高的限制以防计算错误或Z3找到更多配置
    expected_base_configs_N = (len_N + 1) * math.factorial(len_N) if len_N > 0 else 1
    expected_base_configs_P = (len_P + 1) * math.factorial(len_P) if len_P > 0 else 1
    max_base_configs_to_find = expected_base_configs_N * expected_base_configs_P + 10  # 添加安全余量

    # 使用Z3求解器找到所有满足约束的基础配置
    while s.check() == sat:  # 当还有满足约束的解时
        if base_configs_found_count >= max_base_configs_to_find:  # 检查是否达到限制
            print(f"已达到基础配置的内部限制 ({max_base_configs_to_find})。正在停止Z3阶段。")
            break

        m = s.model()  # 获取当前解的模型
        # 提取模型数据
        model_data = {
            'n_assignments': [{'row': m.eval(n_row_vars[i]).as_long(),  # NMOS晶体管的行分配
                               'order': m.eval(n_order_vars[i]).as_long()}  # NMOS晶体管的顺序
                              for i in range(len_N)],
            'p_assignments': [{'row_in_p_group': m.eval(p_row_vars[j]).as_long() - 2,  # PMOS晶体管的行分配（相对于PMOS组）
                               'order': m.eval(p_order_vars[j]).as_long()}  # PMOS晶体管的顺序
                              for j in range(len_P)]
        }
        base_configurations_from_z3.append(model_data)  # 保存基础配置
        base_configs_found_count += 1  # 增加计数

        # 构建排除当前解的约束（以找到下一个不同的解）
        block = []
        for i in range(len_N):  # 排除当前NMOS分配
            block.append(n_row_vars[i] != m.eval(n_row_vars[i]))
            block.append(n_order_vars[i] != m.eval(n_order_vars[i]))
        for i in range(len_P):  # 排除当前PMOS分配
            block.append(p_row_vars[i] != m.eval(p_row_vars[i]))
            block.append(p_order_vars[i] != m.eval(p_order_vars[i]))

        if not block: break  # 如果没有约束可添加，退出循环
        s.add(Or(block))  # 添加排除约束

    print(f"阶段1: Z3 找到 {len(base_configurations_from_z3)} 个基础配置。")

    # --- 阶段2: Python 为每个基础配置扩展方向 ---
    all_intermediate_solutions = []  # 存储所有扩展后的解（去重前）
    if not base_configurations_from_z3:  # 如果Z3没有找到任何基础配置
        print("Z3 未找到任何基础配置。")
        return []

    print(f"阶段2: 正在为 {len(base_configurations_from_z3)} 个基础配置扩展方向...")

    # 计算方向组合的数量
    num_n_orient_combos = int(math.pow(2, len_N))  # NMOS方向组合数：2^len_N
    num_p_orient_combos = int(math.pow(2, len_P))  # PMOS方向组合数：2^len_P

    # 为每个基础配置生成所有可能的方向组合
    for sol_idx, base_config in enumerate(base_configurations_from_z3):
        if sol_idx % 100 == 0 and sol_idx > 0:  # 进度报告
            print(f"  正在处理基础配置 {sol_idx + 1}/{len(base_configurations_from_z3)}...")

        # 遍历所有NMOS方向组合
        for n_orient_combo_idx in range(num_n_orient_combos):
            # 将索引转换为二进制表示的方向列表
            current_n_orientations = [(n_orient_combo_idx >> i) & 1 for i in range(len_N)]

            # 遍历所有PMOS方向组合
            for p_orient_combo_idx in range(num_p_orient_combos):
                # 将索引转换为二进制表示的方向列表
                current_p_orientations = [(p_orient_combo_idx >> i) & 1 for i in range(len_P)]

                # 为当前配置创建4行的解决方案
                current_solution_data_rows = [[] for _ in range(4)]
                row0_details, row1_details = [], []  # NMOS行的详细信息

                if len_N > 0:  # 如果有NMOS晶体管
                    # 为每个NMOS晶体管创建详细信息
                    for i in range(len_N):
                        n_assign = base_config['n_assignments'][i]  # 获取分配信息
                        orientation_is_reversed = bool(current_n_orientations[i])  # 方向是否反转
                        detail = {'trans_info': DataN_transistors[i],  # 晶体管信息
                                  'orientation_is_reversed': orientation_is_reversed,  # 方向信息
                                  'order': n_assign['order']}  # 顺序信息
                        # 根据行分配添加到相应的行
                        if n_assign['row'] == 0:
                            row0_details.append(detail)
                        else:
                            row1_details.append(detail)

                    # 按顺序排序并构建行输出
                    row0_details.sort(key=lambda x: x['order'])
                    row1_details.sort(key=lambda x: x['order'])
                    current_solution_data_rows[0] = build_row_output(row0_details)
                    current_solution_data_rows[1] = build_row_output(row1_details)

                row2_details, row3_details = [], []  # PMOS行的详细信息
                if len_P > 0:  # 如果有PMOS晶体管
                    # 为每个PMOS晶体管创建详细信息
                    for j in range(len_P):
                        p_assign = base_config['p_assignments'][j]  # 获取分配信息
                        orientation_is_reversed = bool(current_p_orientations[j])  # 方向是否反转
                        detail = {'trans_info': DataP_transistors[j],  # 晶体管信息
                                  'orientation_is_reversed': orientation_is_reversed,  # 方向信息
                                  'order': p_assign['order']}  # 顺序信息
                        # 根据行分配添加到相应的行
                        if p_assign['row_in_p_group'] == 0:
                            row2_details.append(detail)
                        else:
                            row3_details.append(detail)

                    # 按顺序排序并构建行输出
                    row2_details.sort(key=lambda x: x['order'])
                    row3_details.sort(key=lambda x: x['order'])
                    current_solution_data_rows[2] = build_row_output(row2_details)
                    current_solution_data_rows[3] = build_row_output(row3_details)

                all_intermediate_solutions.append(current_solution_data_rows)  # 保存当前解

    print(f"阶段2: Python扩展完成。总共生成 {len(all_intermediate_solutions)} 个中间解（去重前）。")

    # --- 阶段3: 去重并应用 max_unique_solutions_count ---
    print(f"阶段3: 正在对 {len(all_intermediate_solutions)} 个中间解进行去重...")
    unique_final_solutions = []  # 存储唯一的最终解
    seen_data_rows_tuples = set()  # 用于去重的集合

    # 遍历所有中间解进行去重
    for idx, data_rows_list in enumerate(all_intermediate_solutions):
        if idx % 10000 == 0 and idx > 0:  # 去重进度报告
            print(f"  去重进度: 已处理 {idx}/{len(all_intermediate_solutions)}...")

        # 将元素转换为字符串以确保可哈希性（特别是处理NaN）
        current_rows_tuple = tuple(tuple(str(element) for element in row) for row in data_rows_list)

        # 检查是否是新的唯一解
        if current_rows_tuple not in seen_data_rows_tuples:
            if len(unique_final_solutions) < max_unique_solutions_count:  # 检查是否达到数量限制
                seen_data_rows_tuples.add(current_rows_tuple)  # 添加到已见集合
                unique_final_solutions.append(data_rows_list)  # 添加到唯一解列表
            else:
                # 已达到唯一解数量上限
                print(f"在去重过程中已达到 max_unique_solutions_count ({max_unique_solutions_count})。")
                break

    print(f"阶段3: 去重完成。找到 {len(unique_final_solutions)} 个唯一的最终解。")
    return unique_final_solutions  # 返回所有唯一的最终解


def write_solutions_to_csv(solutions_data, output_filename="output.csv", max_total_lines=100000):
    """
    将解决方案写入CSV文件
    solutions_data: 解决方案数据列表
    output_filename: 输出文件名
    max_total_lines: 最大行数限制
    """
    lines_written_count = 0  # 已写入的行数计数
    actual_solutions_written = 0  # 实际写入的解决方案数量

    # 打开CSV文件进行写入
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)  # 创建CSV写入器

        # 遍历所有解决方案
        for solution_idx, four_row_solution_set in enumerate(solutions_data):
            # 检查是否会超过行数限制
            if lines_written_count + 4 > max_total_lines and lines_written_count > 0:
                print(f"在写入解 {solution_idx + 1} 前已达到CSV最大行数限制 ({max_total_lines})。停止写入CSV。")
                break

            # 写入当前解决方案的4行数据
            for row_data_list in four_row_solution_set:
                writer.writerow(row_data_list)  # 写入一行数据
            lines_written_count += 4  # 更新已写入行数
            actual_solutions_written += 1  # 更新已写入解决方案数量

            # 检查是否达到行数限制
            if lines_written_count >= max_total_lines:
                print(f"在写入解 {solution_idx + 1} 后已达到CSV最大行数限制 ({max_total_lines})。停止写入CSV。")
                break

    print(f"写入 {lines_written_count} 行 ({actual_solutions_written} 个解) 到 {output_filename}。")


# --- 示例用法 ---
if __name__ == '__main__':
    # 示例数据：每个子列表代表一个晶体管 [源极/漏极1, 源极/漏极2, 栅极]
    DataN_example = [[1, 2, 10], [1, 4, 10], [2, 4, 11]]  # NMOS晶体管数据，L_N = 3
    DataP_example = [[1, 2, 10], [1, 4, 10], [2, 4, 11]]  # PMOS晶体管数据，L_P = 3

    # 复杂度分析示例（对于 L_N=4, L_P=3）:
    # Z3 基础配置数: (4+1)*4! * (3+1)*3! = 120 * 24 = 2880
    # Python 方向扩展数: 2^4 * 2^3 = 16 * 8 = 128
    # 总中间解数（去重前）: 2880 * 128 = 368,640

    print("启动Z3求解过程 (使用Python进行方向扩展和最终去重)...")
    # max_unique_solutions_count 对应最终输出的唯一解数量
    # CSV限制10万行 -> 25000个解（每个解4行）
    found_placements = solve_transistor_placements(DataN_example, DataP_example, max_unique_solutions_count=2500000)

    print(f"找到的总独立布局解数量 (方向扩展和去重后): {len(found_placements)}")

    # 将解决方案写入CSV文件
    write_solutions_to_csv(found_placements, "output.csv", max_total_lines=10000000)
