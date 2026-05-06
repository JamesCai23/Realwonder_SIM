from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs
import math
from simulation.utils import gs_to_pt3d

@register_case("tree")
class Tree(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)
    

    def fix_particles(self):
        ############################ Fix Particles ###############################
        # fix part of the object - only applied to particle cases
       
        fixed_area_list = list(self.config['fixed_area'])[0]
        sim_particles = torch.tensor(self.all_objs[0].init_particles).to(self.device)
        x_left = self.all_obj_info[0]['min'][0] + self.all_obj_info[0]['size'][0] * fixed_area_list[0]
        x_right = self.all_obj_info[0]['min'][0] + self.all_obj_info[0]['size'][0] * fixed_area_list[1]
        z_top = self.all_obj_info[0]['max'][2] - self.all_obj_info[0]['size'][2] * fixed_area_list[2]
        z_bottom = self.all_obj_info[0]['max'][2] - self.all_obj_info[0]['size'][2] * fixed_area_list[3]

        fixed_area_idx = torch.where(
            (sim_particles[:, 0] > x_left) & (sim_particles[:, 0] < x_right) &
            (sim_particles[:, 2] > z_bottom) & (sim_particles[:, 2] < z_top)
        )

        is_free = torch.ones(sim_particles.shape[0], dtype=torch.bool).to(self.device)
        is_free[fixed_area_idx] = False
        self.all_objs[0].set_free(is_free)
    
    
    def create_force_fields(self):

        wind_force_center = (
            self.all_obj_info[0]['min'][0].item(),
            self.all_obj_info[0]['min'][1].item(),
            self.all_obj_info[0]['min'][2].item()
        )
        wind_force_radius = ((self.all_obj_info[0]['max'][0] - self.all_obj_info[0]['min'][0]) * 1.5).item()
        wind_force_direction = (
            ((self.all_obj_info[0]['max'][0] - self.all_obj_info[0]['min'][0]) * 0.3).item(),
            ((self.all_obj_info[0]['min'][1] - self.all_obj_info[0]['max'][1]) * 0.).item(),
            ((self.all_obj_info[0]['min'][2] - self.all_obj_info[0]['max'][2]) * 0.0).item()
        )
        force_field = gs.force_fields.Wind(
            direction = wind_force_direction,
            strength = 1,
            radius = wind_force_radius,
            center = wind_force_center,
        )
        force_field.activate()
        self.scene.add_force_field(
            force_field = force_field
        )
