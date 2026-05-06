from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("santa_cloth")
class SantaCloth(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        pass

    def fix_particles(self):
        ############################ Fix Particles ###############################
        # fix part of the object - only applied to particle cases
        for i in range(len(self.all_objs)):
            fixed_area_list = list(self.config['fixed_area'])[i]
            sim_particles = torch.tensor(self.all_objs[i].init_particles).to(self.device)
            x_left = self.all_obj_info[i]['min'][0] + self.all_obj_info[i]['size'][0] * fixed_area_list[0]
            x_right = self.all_obj_info[i]['min'][0] + self.all_obj_info[i]['size'][0] * fixed_area_list[1]
            z_top = self.all_obj_info[i]['max'][2] - self.all_obj_info[i]['size'][2] * fixed_area_list[2]
            z_bottom = self.all_obj_info[i]['max'][2] - self.all_obj_info[i]['size'][2] * fixed_area_list[3]

            print(f"[fix_particles] obj {i}: fixed_area={fixed_area_list}, "
                  f"n_particles={len(sim_particles)}")
            print(f"[fix_particles] obj {i}: x_range=[{x_left:.4f}, {x_right:.4f}], "
                  f"z_range=[{z_bottom:.4f}, {z_top:.4f}]")
            print(f"[fix_particles] obj {i}: particle x range=[{sim_particles[:,0].min():.4f}, "
                  f"{sim_particles[:,0].max():.4f}], "
                  f"z range=[{sim_particles[:,2].min():.4f}, {sim_particles[:,2].max():.4f}]")
            print(f"[fix_particles] obj {i}: obj_info min={self.all_obj_info[i]['min']}, "
                  f"max={self.all_obj_info[i]['max']}, size={self.all_obj_info[i]['size']}")

            fixed_area_idx = torch.where(
                (sim_particles[:, 0] > x_left) & (sim_particles[:, 0] < x_right) &
                (sim_particles[:, 2] > z_bottom) & (sim_particles[:, 2] < z_top)
            )

            fixed_area_points = sim_particles[fixed_area_idx]
            print(f"[fix_particles] obj {i}: found {len(fixed_area_points)} particles to fix")
            fixed_area_list = [tuple(point.tolist()) for point in fixed_area_points]
            for point in fixed_area_list:
                self.all_objs[i].fix_particle(self.all_objs[i].find_closest_particle(point), 0)
            print(f"[fix_particles] obj {i}: fixed {len(fixed_area_list)} particles")

    def create_force_fields(self):
        if self.config.get('skip_force_fields', False):
            return

        wind_lowest = (self.all_obj_occupied_lower_bound[2] + (self.all_obj_occupied_upper_bound[2] - self.all_obj_occupied_lower_bound[2]) * 0.05).cpu().numpy()
        wind_highest = (self.all_obj_occupied_lower_bound[2] + (self.all_obj_occupied_upper_bound[2] - self.all_obj_occupied_lower_bound[2]) * 0.8).cpu().numpy()

        @ti.func
        def force_func(pos, vel, t, i):
            frame_step = t // self.config['dt'] // self.config['frame_steps']
            strength = 3
            direction = ti.Vector([0, 0, 0], dt=gs.ti_float)

            cycle_step = frame_step % 144
            if cycle_step <= 20:
                direction = ti.Vector([-1, 0, 0], dt=gs.ti_float)
            elif cycle_step > 20 and cycle_step <= 30:
                direction = ti.Vector([0, 0, 0], dt=gs.ti_float)
            elif cycle_step > 30 and cycle_step <= 50:
                direction = ti.Vector([1, 0, 0], dt=gs.ti_float)
            elif cycle_step > 50 and cycle_step <= 80:
                direction = ti.Vector([-1, 0, 0], dt=gs.ti_float)
                strength = 3
            elif cycle_step > 80 and cycle_step <= 100:
                direction = ti.Vector([0, 0, 0], dt=gs.ti_float)
                strength = 0
            elif cycle_step > 100 and cycle_step <= 120:
                direction = ti.Vector([1, 0, 0], dt=gs.ti_float)
                strength = 3
            elif cycle_step > 120 and cycle_step <= 130:
                direction = ti.Vector([0, 0, 0], dt=gs.ti_float)
                strength = 0
            elif cycle_step > 130 and cycle_step <= 144:
                direction = ti.Vector([-1, 0, 0], dt=gs.ti_float)
                strength = 3

            acc = ti.Vector.zero(gs.ti_float, 3)
            if pos[2] > wind_lowest and pos[2] < wind_highest:
                scaler = (pos[2] - wind_lowest) / (wind_highest - wind_lowest)
                scaler = ti.exp(scaler ** 2)
                acc = direction * strength * scaler
            else:
                acc = ti.Vector.zero(gs.ti_float, 3)
            return acc

        force_field = gs.force_fields.Custom(force_func)
        force_field.activate()
        self.scene.add_force_field(
            force_field = force_field
        )
    
    def detect_ground_plane(self, ground_plane):
        pass
