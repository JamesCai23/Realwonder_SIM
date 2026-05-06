from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs
import math
from simulation.utils import gs_to_pt3d

@register_case("sand_house")
class SandHouse(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)
        self.arm_scale = 1

    def add_robots(self):
        cup_pos = self.all_obj_info[0]['center'].cpu().numpy()
        
        franka_pos1 = cup_pos + np.array([-0.65, -0.1, 0.0])
        franka_pos2 = cup_pos + np.array([0.65, -0.1, 0.0])
        franka_pos1[2] = self.all_obj_info[0]['min'][2]
        franka_pos2[2] = self.all_obj_info[0]['min'][2]
        self.franka_pos1 = torch.from_numpy(franka_pos1).to(self.device)
        self.franka_pos2 = torch.from_numpy(franka_pos2).to(self.device)
        self.franka1 = self.scene.add_entity(
            gs.morphs.MJCF(
                file ="cases/xml/franka_emika_panda/panda.xml",
                pos = franka_pos1,
                euler =(0, 0, 90),
                scale = self.arm_scale
            )
        )
        self.franka2 = self.scene.add_entity(
            gs.morphs.MJCF(
                file="cases/xml/franka_emika_panda/panda.xml",
                pos= franka_pos2,
                euler=(0, 0, -90),
                scale = self.arm_scale
            )
        )
        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
    
    def fix_particles(self):
        self.init_pos = self.all_objs[0].get_particles()[0]
    
    def init_robots_pose(self):

        cup_pos = self.all_obj_info[0]['center'].cpu().numpy()

        self.initial_ee_pos1 = cup_pos + np.array([-0.5, -0.1, -0.03])
        self.initial_ee_pos2 = cup_pos + np.array([0.45, -0.1, -0.09])
        initial_ee_quat1 = np.array([0, 0, 0.7071, 0.7071])
        initial_ee_quat2 = np.array([0.7071, 0, -0.7071, 0])

        self.end_effector1 = self.franka1.get_link("hand")
        self.end_effector2 = self.franka2.get_link("hand")
        
        qpos1 = self.franka1.inverse_kinematics(
            link=self.end_effector1,
            pos=self.initial_ee_pos1,
            quat=initial_ee_quat1,
        )
        qpos2 = self.franka2.inverse_kinematics(
            link=self.end_effector2,
            pos=self.initial_ee_pos2,
            quat=initial_ee_quat2,
        )

        if qpos1 is not None:
            self.franka_qpos1 = qpos1.cpu().numpy()
            self.franka_qpos1[-2:] = 0.04 * self.arm_scale
        else:
            raise ValueError("IK failed")

        if qpos2 is not None:
            self.franka_qpos2 = qpos2.cpu().numpy()
            self.franka_qpos2[-2:] = 0.04 * self.arm_scale
        else:
            raise ValueError("IK failed")

        self.franka1.set_qpos(self.franka_qpos1)
        self.franka2.set_qpos(self.franka_qpos2)

    def custom_simulation(self, sid):

        frame_num = sid // self.config['frame_steps']
        if sid == 0:
            self.franka1.set_dofs_kp(self.franka1.get_dofs_kp() * 5)
            force_min, force_max = self.franka1.get_dofs_force_range()
            self.franka1.set_dofs_force_range(lower=force_min * 5, upper=force_max * 5)
            self.franka2.set_dofs_kp(self.franka2.get_dofs_kp() * 5)
            force_min, force_max = self.franka2.get_dofs_force_range()
            self.franka2.set_dofs_force_range(lower=force_min * 5, upper=force_max * 5)
 

        if frame_num < 21:
            self.all_objs[0].set_position(self.init_pos)

        step_size = 0.018
        down_size = 0.002
        target_quat = np.array([0, 0, 0.7071, 0.7071])


        if frame_num <= 28:
            self.franka2.set_qpos(self.franka_qpos2)
            self.active_franka = self.franka1

            progress = frame_num / 28.0
            displacement = progress * 28 * step_size 
            target_pos = self.initial_ee_pos1.copy() + np.array([displacement, 0.0, 0.0])
            
            self.franka_qpos1 = self.franka1.inverse_kinematics(
                link=self.end_effector1,
                pos=target_pos,
                # quat=target_quat,
            )
            
            if self.franka_qpos1 is not None:
                self.franka_qpos1[..., -2:] = 0.04
                self.franka1.control_dofs_position(self.franka_qpos1[..., :-2], self.motors_dof)
                self.franka1.control_dofs_position(np.array([0.04, 0.04]), self.fingers_dof)
        
        elif 28 < frame_num <= 40:
            self.franka2.set_qpos(self.franka_qpos2)
            self.active_franka = self.franka1
            progress = (frame_num - 28) / 12.0
            retreat_distance = progress * 28 * step_size
            target_pos = self.initial_ee_pos1.copy() + np.array([28 * step_size - retreat_distance, 0.0, 0.0])
            
            self.franka_qpos1 = self.franka1.inverse_kinematics(
                link=self.end_effector1,
                pos=target_pos,
                # quat=target_quat,
            )
            
            if self.franka_qpos1 is not None:
                self.franka_qpos1[..., -2:] = 0.04
                self.franka1.control_dofs_position(self.franka_qpos1[..., :-2], self.motors_dof)
                self.franka1.control_dofs_position(np.array([0.04, 0.04]), self.fingers_dof)
        
        elif 40 < frame_num <= 70:

            self.franka1.set_qpos(self.franka_qpos1)
            self.active_franka = self.franka2
            target_quat = np.array([0.7071, 0, -0.7071, 0])

            progress = (frame_num - 40) / 30.0
            displacement = progress * 30 * step_size
            target_pos = self.initial_ee_pos2.copy() + np.array([-displacement, 0.0, 0.0])
            
            self.franka_qpos2 = self.franka2.inverse_kinematics(
                link=self.end_effector2,
                pos=target_pos,
                quat=target_quat,
            )
            
            if self.franka_qpos2 is not None:
                self.franka_qpos2[..., -2:] = 0.04
                self.franka2.control_dofs_position(self.franka_qpos2[..., :-2], self.motors_dof)
                self.franka2.control_dofs_position(np.array([0.04, 0.04]), self.fingers_dof)
        
        elif 70 < frame_num <= 100:

            self.franka1.set_qpos(self.franka_qpos1)
            self.active_franka = self.franka2
            target_quat = np.array([0.7071, 0, -0.7071, 0])

            progress = (frame_num - 70) / 30.0
            retreat_distance = progress * 30 * step_size
            target_pos = self.initial_ee_pos2.copy() + np.array([-30 * step_size + retreat_distance, 0.0, 0.0])
            
            self.franka_qpos2 = self.franka2.inverse_kinematics(
                link=self.end_effector2,
                pos=target_pos,
                quat=target_quat,
            )
            
            if self.franka_qpos2 is not None:
                self.franka_qpos2[..., -2:] = 0.04
                self.franka2.control_dofs_position(self.franka_qpos2[..., :-2], self.motors_dof)
                self.franka2.control_dofs_position(np.array([0.04, 0.04]), self.fingers_dof)
        
        elif 100 < frame_num <= 130:
            self.franka2.set_qpos(self.franka_qpos2)
            self.active_franka = self.franka1

            progress = (frame_num - 100) / 30.0
            displacement = progress * 30 * step_size
            target_pos = self.initial_ee_pos1.copy() + np.array([displacement, 0.1, 0.0])
            
            self.franka_qpos1 = self.franka1.inverse_kinematics(
                link=self.end_effector1,
                pos=target_pos,
                # quat=target_quat,
            )
            
            if self.franka_qpos1 is not None:
                self.franka_qpos1[..., -2:] = 0.04
                self.franka1.control_dofs_position(self.franka_qpos1[..., :-2], self.motors_dof)
                self.franka1.control_dofs_position(np.array([0.04, 0.04]), self.fingers_dof)
        
        elif 130 < frame_num <= 160:
            self.franka2.set_qpos(self.franka_qpos2)
            self.active_franka = self.franka1
            progress = (frame_num - 130) / 30.0
            retreat_distance = progress * 30 * step_size
            target_pos = self.initial_ee_pos1.copy() + np.array([30 * step_size - retreat_distance, 0.1, 0.0])
            
            self.franka_qpos1 = self.franka1.inverse_kinematics(
                link=self.end_effector1,
                pos=target_pos,
                # quat=target_quat,
            )
            
            if self.franka_qpos1 is not None:
                self.franka_qpos1[..., -2:] = 0.04
                self.franka1.control_dofs_position(self.franka_qpos1[..., :-2], self.motors_dof)
                self.franka1.control_dofs_position(np.array([0.04, 0.04]), self.fingers_dof)
        
        elif frame_num > 160:
            self.franka1.set_qpos(self.franka_qpos1)
            self.franka2.set_qpos(self.franka_qpos2)
            self.active_franka = self.franka1
            
    def after_simulation_step(self, svr):
        franka_vertices, franka_faces, franka_colors = self.extract_franka_mesh_data_combined(self.active_franka)
        franka_vertices = gs_to_pt3d(franka_vertices)
        if svr.franka_mesh is None:
            svr.franka_mesh = {
                'vertices': franka_vertices,
                'faces': franka_faces,
                'colors': franka_colors
            }
        else:
            svr.franka_mesh['vertices'] = franka_vertices