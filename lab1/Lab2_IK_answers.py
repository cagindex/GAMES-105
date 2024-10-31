import numpy as np
from scipy.spatial.transform import Rotation as R

def get_off_orientation(p1: np.ndarray, p2: np.ndarray, scale: float = 1.0) -> R:
    # Get the orientation quat from p1 -> p2 i.e. p2 = quat * p1
    # Scale: scales the angle
    assert np.linalg.norm(p1) != 0.0
    assert np.linalg.norm(p2) != 0.0

    normalized_p1 = p1 / np.linalg.norm(p1)
    normalized_p2 = p2 / np.linalg.norm(p2)

    if np.allclose(normalized_p1, normalized_p2):
        return R.from_quat([0.0, 0.0, 0.0, 1.0])

    dot_product = np.dot(normalized_p1, normalized_p2)
    cross_product = np.cross(normalized_p1, normalized_p2)

    rotate_axis = cross_product
    rotate_axis_norm = np.linalg.norm(rotate_axis)

    normalized_rotate_axis = rotate_axis / rotate_axis_norm
    angle = scale * np.arccos(np.clip(dot_product, -1.0, 1.0))

    return R.from_rotvec(normalized_rotate_axis * angle)


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose, threshold=0.01):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    max_iter = 300
    root_joint_idx = meta_data.joint_name.index(meta_data.root_joint)
    end_joint_idx = meta_data.joint_name.index(meta_data.end_joint)
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()

    flag = False
    for _ in range(max_iter):
        if flag:
            print(f"FIND SOLUTION WITH ITERATION {_}")
            break
        end_pose = joint_positions[end_joint_idx]
        for _idx_ in range(len(path)-2, -1, -1):
            if np.linalg.norm(joint_positions[end_joint_idx] - target_pose) <= threshold :
                flag = True
                break
            idx = path[_idx_]
            if idx == 0:
                continue
            selected_pose = joint_positions[idx]
            rotation = get_off_orientation(end_pose - selected_pose, target_pose - selected_pose, 0.5)
            # Update all related nodes
            for _i_ in range(_idx_+1, len(path)):
                i = path[_i_]
                node_pose = joint_positions[i]
                new_node_pose = rotation.apply(node_pose - selected_pose) + selected_pose
                joint_positions[i] = new_node_pose
    
    # Update joint orientation
    prev_joint_orientation = joint_orientations.copy()
    for _idx_ in range(1, len(path1)):
        idx, child_idx = path1[_idx_], path1[_idx_-1]
        after_local_offset = joint_positions[child_idx] - joint_positions[idx] 
        initi_local_offset = meta_data.joint_initial_position[child_idx] - meta_data.joint_initial_position[idx]

        rotation = get_off_orientation(initi_local_offset, after_local_offset) 

        joint_orientations[idx] = rotation.as_quat()
    for _idx_ in range(0, len(path2)):
        if _idx_+1 == len(path2):
            if path2[_idx_] == 0:
                break
            idx, father_idx = path1[-1], path2[_idx_]
        else:
            idx, father_idx = path2[_idx_], path2[_idx_+1] 
        after_local_offset = joint_positions[idx] - joint_positions[father_idx]
        initi_local_offset = meta_data.joint_initial_position[idx] - meta_data.joint_initial_position[father_idx]

        rotation = get_off_orientation(initi_local_offset, after_local_offset)

        joint_orientations[father_idx] = rotation.as_quat()

    # Forward ik to update joint global positions
    mem = set(path)
    for idx in range(len(meta_data.joint_parent)):
        father_idx = meta_data.joint_parent[idx]
        if idx in mem or father_idx == -1:
            continue
        initi_local_offset = meta_data.joint_initial_position[idx] - meta_data.joint_initial_position[father_idx]

        rotation = R.from_quat(joint_orientations[idx])
        father_rotation = R.from_quat(joint_orientations[father_idx])
        prev_father_rotation = R.from_quat(prev_joint_orientation[father_idx])

        joint_positions[idx] = father_rotation.apply(initi_local_offset) + joint_positions[father_idx]
        joint_orientations[idx] = (rotation * prev_father_rotation.inv() * father_rotation).as_quat()
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    meta_data.root_joint = 'lShoulder'
    meta_data.end_joint  = "lWrist_end"
    ROOT_offset = np.array([relative_x, 0, relative_z])
    target_pose = joint_positions[0] + ROOT_offset
    target_pose[1] = target_height

    joint_positions, joint_orientations = part1_inverse_kinematics(
        meta_data=meta_data,
        joint_positions=joint_positions,
        joint_orientations=joint_orientations,
        target_pose=target_pose
    )
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    threshold = 0.01
    left_end_name  = 'lWrist_end'
    right_end_name = 'rWrist_end'
    
    left_end_idx  = meta_data.joint_name.index(left_end_name)
    right_end_idx = meta_data.joint_name.index(right_end_name)

    t = 0
    end_names = [left_end_name, right_end_name]
    target_poses = [left_target_pose, right_target_pose]
    while np.linalg.norm(joint_positions[left_end_idx] - left_target_pose) > threshold or \
          np.linalg.norm(joint_positions[right_end_idx] - right_target_pose) > threshold :
        meta_data.end_joint = end_names[t]
        joint_positions, joint_orientations = part1_inverse_kinematics(
            meta_data=meta_data,
            joint_positions=joint_positions,
            joint_orientations=joint_orientations,
            target_pose=target_poses[t],
            threshold=0.0005
        )
        t = 1 - t
    return joint_positions, joint_orientations