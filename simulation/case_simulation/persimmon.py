from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("persimmon")
class Persimmon(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)
       
    def custom_simulation(self, sid):
        # frame_step = sid // self.config['frame_steps']
        force_direction_left = np.array([-1.0, 0.0, 0.0])
        force_direction_right = np.array([1.0, 0.0, 0.0])
        left_force_strength = 100
        right_force_strength = 100

        force_right = (right_force_strength * force_direction_right / np.linalg.norm(force_direction_right)).reshape(1, 3)
        force_left = (left_force_strength * force_direction_left / np.linalg.norm(force_direction_left)).reshape(1, 3)

        if sid <= 5:
            self.all_objs[1].solver.apply_links_external_force(force=force_left, links_idx=[self.all_objs[1].idx])
            self.all_objs[0].solver.apply_links_external_force(force=force_right, links_idx=[self.all_objs[0].idx])

        return None
    
    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        self.all_obj_info[0]['center'][2] += 0.025
        self.all_obj_info[1]['center'][2] += 0.015
        # import pdb; pdb.set_trace()
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)