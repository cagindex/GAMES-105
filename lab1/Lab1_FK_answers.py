import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []

    joint_idx_stack = [] 
    with open(bvh_file_path, 'r') as f:
        f.readline() # pass HIERARCHY section
        while (line_content := f.readline().strip()) != 'MOTION':
            if line_content.startswith('ROOT'):
                joint_parent.append(-1)
                joint_idx_stack.append(0)
                joint_name.append(line_content.split()[1])
            elif line_content.startswith('JOINT'):
                joint_parent.append(joint_idx_stack[-1])
                joint_idx_stack.append(len(joint_name))
                joint_name.append(line_content.split()[1])
            elif line_content.startswith('End Site'):
                joint_parent.append(joint_idx_stack[-1])
                joint_idx_stack.append(len(joint_name))
                joint_name.append(joint_name[-1] + '_end')
            elif line_content.startswith('OFFSET'):
                joint_offset.append(list(map(float, line_content.split()[1:])))
            elif line_content.startswith('}'):
                joint_idx_stack.pop()
            else:
                continue
    joint_offset = np.array(joint_offset, dtype=np.float32)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = []
    joint_orientations = []

    idx = -1
    data = motion_data[frame_id].reshape(-1, 3)
    for i in range(len(joint_name)):
        idx += 0 if joint_name[i].endswith('_end') else 1
        parent_idx, offset = joint_parent[i], joint_offset[i]
        if parent_idx == -1:
            joint_positions.append(data[idx])
            joint_orientations.append(R.from_euler('XYZ', data[idx+1], degrees=True).as_quat())
        else:
            parent_orientation = joint_orientations[parent_idx]
            parent_position = joint_positions[parent_idx]

            joint_position = parent_position + R.from_quat(parent_orientation).apply(offset)
            joint_orientation = (R.from_quat(parent_orientation) * R.from_euler('XYZ', data[idx+1], degrees=True)).as_quat()

            joint_positions.append(joint_position)
            joint_orientations.append(joint_orientation)

    joint_positions = np.array(joint_positions, dtype=np.float32)
    joint_orientations = np.array(joint_orientations, dtype=np.float32)
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    def get_global_joint_pos(joint_idx , joint_parent, joint_offset):
        pos = np.array([0.0, 0.0, 0.0], dtype=joint_offset.dtype)
        while (joint_idx != -1):
            pos += joint_offset[joint_idx]
            joint_idx = joint_parent[joint_idx]
        return pos
    
    def get_off_orientation(p1: np.ndarray, p2: np.ndarray):
        # Get the orientation quat from p1 -> p2 i.e. p2 = quat * p1
        normalized_p1 = p1 / np.linalg.norm(p1)
        normalized_p2 = p2 / np.linalg.norm(p2)

        if np.allclose(normalized_p1, normalized_p2):
            return R.from_quat([0.0, 0.0, 0.0, 1.0])

        dot_product = np.dot(normalized_p1, normalized_p2)
        cross_product = np.cross(normalized_p1, normalized_p2)

        rotate_axis = cross_product
        rotate_axis_norm = np.linalg.norm(rotate_axis)

        normalized_rotate_axis = rotate_axis / rotate_axis_norm
        angle = np.arccos(dot_product)

        return R.from_rotvec(normalized_rotate_axis * angle)

    def find_next_idx(name, joint_names, joint_parent):
        count = 0
        idx = -1
        for i in range(len(joint_names)):
            if joint_names[joint_parent[i]] == name:
                idx = i
                count += 1
        if count == 1:
            return idx
        else:
            return -1

    motion_data = load_motion_data(A_pose_bvh_path)
    ret_motion_data = np.zeros_like(motion_data)
    t_joint_name, t_joint_parent, t_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    a_joint_name, a_joint_parent, a_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)

    a_joint_idx, count = {}, 0
    for i in range(len(a_joint_name)):
        if a_joint_name[i].endswith('_end'):
            continue
        a_joint_idx[a_joint_name[i]] = count
        count += 1
    t_joint_idx, count = {}, 0
    for i in range(len(t_joint_name)):
        if t_joint_name[i].endswith('_end'):
            continue
        t_joint_idx[t_joint_name[i]] = count
        count += 1

    for name in t_joint_name:
        if name.endswith('_end'):
            continue
        t_idx = t_joint_name.index(name)
        a_idx = a_joint_name.index(name)

        a_idx_next = find_next_idx(name, a_joint_name, a_joint_parent)
        t_idx_next = find_next_idx(name, t_joint_name, t_joint_parent)
        a_idx_next = a_idx_next if a_idx_next != -1 else a_idx
        t_idx_next = t_idx_next if t_idx_next != -1 else t_idx

        t_global_joint_pose = get_global_joint_pos(t_idx_next, t_joint_parent, t_joint_offset)
        a_global_joint_pose = get_global_joint_pos(a_idx_next, a_joint_parent, a_joint_offset)

        a_data_idx = a_joint_idx[name] 
        t_data_idx = t_joint_idx[name] 
        data = R.from_euler('XYZ', motion_data[:, 3+3*a_data_idx:3+3*(a_data_idx+1)], degrees=True)
        if not np.allclose(t_global_joint_pose, a_global_joint_pose) :
            print(name)
            r_off = get_off_orientation(a_joint_offset[a_idx], t_joint_offset[t_idx])
            r_off_next = get_off_orientation(a_joint_offset[a_idx_next], t_joint_offset[t_idx_next])

            data = r_off * data * r_off_next.inv()
        ret_motion_data[:, 3+3*t_data_idx:3+3*(t_data_idx+1)] = data.as_euler('XYZ', degrees=True)
    
    ret_motion_data[:, :3] = motion_data[:, :3]
    return ret_motion_data
