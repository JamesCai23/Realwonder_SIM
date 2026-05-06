from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("lamp")
class Lamp(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)
    
    def custom_simulation(self, sid):
        force_direction = np.array(self.config.get('force_direction', [1.0, 0.0, 0.0]))
        force_direction = force_direction / np.linalg.norm(force_direction)
        force_direction = force_direction.reshape(1, 3)
        force_strength = self.config.get('force_strength', 1.0)
        # self.all_objs[0].solver.set_geoms_friction_ratio(friction_ratio=[0.01], geoms_idx=[self.all_objs[0].geoms[0].idx])
        self.all_objs[0].solver.apply_links_external_force(force=force_direction * force_strength, links_idx=[self.all_objs[0].idx])    
        
        return None