from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("two_duck")
class TwoDuck(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)
        self.initial_pos0 = all_obj_info[0]['center'].cpu().numpy()
        self.initial_pos1 = all_obj_info[1]['center'].cpu().numpy()

    def custom_simulation(self, sid):
        frame_step = sid // self.config['frame_steps']
        pos_0 = self.all_objs[0].get_pos().cpu().numpy()
        pos_1 = self.all_objs[1].get_pos().cpu().numpy()
        force_direction = pos_1 - pos_0
        force_direction = force_direction / np.linalg.norm(force_direction)
        force_direction = force_direction.reshape(1, 3)
        force_strength = 0.2
        
        if sid == 0:
            self.all_objs[0].solver.apply_links_external_torque(torque=np.array([[0, 0, -0.01]]), links_idx=[self.all_objs[0].idx])

        # self.all_objs[0].solver.set_geoms_friction_ratio(friction_ratio=[0.01], geoms_idx=[self.all_objs[0].geoms[0].idx])
        self.all_objs[0].solver.apply_links_external_force(force=force_direction * force_strength, links_idx=[self.all_objs[0].idx])    

        return None