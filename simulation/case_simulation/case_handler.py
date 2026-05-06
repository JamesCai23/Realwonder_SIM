"""
Base Case Handler Template
Abstract base class for all simulation case handlers.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
import gstaichi as ti
import genesis as gs
import sys

CASE_REGISTRY = {}

def register_case(case_name: str):
    """
    A decorator to automatically register the CaseHandler subclass to CASE_REGISTRY.
    """
    def decorator(cls):
        if case_name in CASE_REGISTRY:
            raise ValueError(f"Case name '{case_name}' already registered!")
        
        # Register: map the string case_name to the actual Class Object
        CASE_REGISTRY[case_name] = cls
        print(f"Registered Case: '{case_name}' -> {cls.__name__}")
        return cls # Return the unmodified class
    return decorator

class CaseHandler(ABC):
    """
    Abstract base class for handling case-specific simulation logic.
    Each simulation case should inherit from this class.
    """
    
    def __init__(self, config, all_obj_info: list[dict], device: torch.device):
        self.config = config
        self.all_obj_info = all_obj_info
        self.device = device

    def set_simulation_bounds(self, all_obj_occupied_lower_bound, all_obj_occupied_upper_bound):
        self.all_obj_occupied_lower_bound = all_obj_occupied_lower_bound
        self.all_obj_occupied_upper_bound = all_obj_occupied_upper_bound
        self.all_obj_occupied_size = self.all_obj_occupied_upper_bound - self.all_obj_occupied_lower_bound
        self.simulation_lower_bound = self.all_obj_occupied_lower_bound - 3 * self.all_obj_occupied_size
        self.simulation_upper_bound = self.all_obj_occupied_upper_bound + 3 * self.all_obj_occupied_size

    def get_simulation_bounds(self):
        return self.simulation_lower_bound.cpu().numpy(), self.simulation_upper_bound.cpu().numpy()
    

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        self.obj_materials = obj_materials
        self.obj_vis_modes = obj_vis_modes
        self.scene = scene
        self.objs = []
        if 'is_obj_fixed' not in self.config:
            is_obj_fixed = [False] * len(self.all_obj_info)
        else:
            is_obj_fixed = self.config['is_obj_fixed']
        for idx, per_obj_info in enumerate(self.all_obj_info):
            if "use_primitive" in self.config and self.config['use_primitive']:

                primitive_morhph = gs.morphs.Box(
                        pos=self.all_obj_info[idx]['center'].cpu().numpy().astype(np.float64),
                        size=self.all_obj_info[idx]['size'].cpu().numpy().astype(np.float64),
                        visualization=True,
                        collision=True,
                        fixed=False,
                    )
                per_obj = self.scene.add_entity(
                    material = self.obj_materials[idx],
                    morph = primitive_morhph,
                    surface = gs.surfaces.Default(
                        color = tuple(np.random.rand(3).tolist() + [1.0]),
                        vis_mode = self.obj_vis_modes[idx],
                    ),
                )
            else:
                try:
                    morph = gs.morphs.Mesh(
                            file = per_obj_info['mesh_path'],
                            scale = 1.0,
                            pos = tuple(per_obj_info['center'].cpu().numpy().astype(np.float64)),
                            euler = (0.0, 0.0, 0.0),
                            fixed = is_obj_fixed[idx],
                            # decimate = self.config['decimate'],
                            # convexify = self.config['convexify'],
                        )
                    per_obj = self.scene.add_entity(
                        material = self.obj_materials[idx],
                        morph = morph,
                        # morph = gs.morphs.Box(
                        #     pos = per_obj_info['center'].cpu().numpy(),
                        #     size = per_obj_info['size'].cpu().numpy(),
                        # ),
                        surface = gs.surfaces.Default(
                            color = tuple(np.random.rand(3).tolist() + [1.0]),
                            vis_mode = self.obj_vis_modes[idx],
                        ),
                    )
                except Exception as e:
                    print(e)
                    import pdb; pdb.set_trace()
                    print("trying to add primitive mesh for object", idx)
                    primitive_morhph = gs.morphs.Box(
                        pos=self.all_obj_info[idx]['center'].cpu().numpy().astype(np.float64),
                        size=self.all_obj_info[idx]['size'].cpu().numpy().astype(np.float64),
                        visualization=True,
                        collision=True,
                        fixed=False,
                    )
                    per_obj = self.scene.add_entity(
                        material = self.obj_materials[idx],
                        morph = primitive_morhph,
                        surface = gs.surfaces.Default(
                            color = tuple(np.random.rand(3).tolist() + [1.0]),
                            vis_mode = self.obj_vis_modes[idx],
                        ),
                    )
            self.objs.append(per_obj)
    
        return self.objs



    
    def before_scene_building(self, scene, all_objs, ground_plane):
        self.scene = scene
        self.all_objs = all_objs
        self.detect_ground_plane(ground_plane)
        self.create_force_fields()
        self.add_robots()
        self.custom_setup()
        self.add_emitters()

    def after_scene_building(self):
        self.init_robots_pose()
        self.fix_particles()
        self.init_obj_states()

    def custom_simulation(self, sid):
        pass

    def after_simulation_step(self, svr):
        pass

    def add_emitters(self):
        """Add emitters if needed for this case."""
        pass

    ## before scene building
    def detect_ground_plane(self, ground_plane):
        """Detect ground plane specific to this case."""
        self.ground_anchor = self.all_obj_occupied_lower_bound.cpu().numpy().copy()
        auto_ground_z = self.config.get('auto_ground_z', None)
        if auto_ground_z is not None:
            clearance = float(self.config.get('ground_clearance', 0.05))
            self.ground_anchor[2] = float(auto_ground_z - clearance)
        else:
            clearance = float(self.config.get('ground_clearance', 0.05))
            self.ground_anchor[2] = float(self.ground_anchor[2] - clearance)
        self.ground_anchor[2] = self.ground_anchor[2]
        self.normal = np.array([0, 0, 1])
        self.scene.add_entity(
            material = gs.materials.Rigid(
                rho = 1000.0 if 'plane_rho' not in self.config else self.config['plane_rho'],
                friction = 5 if 'plane_friction' not in self.config else self.config['plane_friction'],
                coup_friction = 5.0 if 'plane_coup_friction' not in self.config else self.config['plane_coup_friction'],
                coup_softness = 0.002 if 'plane_coup_softness' not in self.config else self.config['plane_coup_softness'],
            ),
            morph = gs.morphs.Plane(pos=(self.ground_anchor[0], self.ground_anchor[1], self.ground_anchor[2]), normal=self.normal)
        )
    
    def create_force_fields(self):
        """Create case-specific force fields."""
        pass
    
    def custom_setup(self):
        """Custom setup for this case."""
        pass
    
    def add_robots(self):
        """Setup robots if needed for this case."""
        pass
    

    ## after scene building
    def init_robots_pose(self):
        """Initialize robots pose if needed for this case."""
        pass

    def fix_particles(self):
        """Fix particles if needed for this case."""
        pass

    def init_obj_states(self):
        """Initialize object states if needed for this case."""
        position_offset = self.config.get('initial_position_offset', None)
        if position_offset is not None:
            if len(position_offset) == 3:
                offset = np.array(position_offset, dtype=np.float64)
                for obj in self.all_objs:
                    current_pos = obj.get_pos().cpu().numpy().astype(np.float64)
                    obj.set_pos(current_pos + offset)
            elif len(position_offset) == len(self.all_objs):
                for idx, obj in enumerate(self.all_objs):
                    per_offset = np.array(position_offset[idx], dtype=np.float64)
                    if per_offset.shape[0] != 3:
                        raise ValueError('Each entry in initial_position_offset must have 3 values: [dx, dy, dz].')
                    current_pos = obj.get_pos().cpu().numpy().astype(np.float64)
                    obj.set_pos(current_pos + per_offset)
            else:
                raise ValueError('initial_position_offset must be [dx, dy, dz] or a list with one [dx, dy, dz] per object.')
        
        # 设置初速度
        if self.config.get('initial_velocity', None) is not None:
            self.all_objs[0].set_dofs_velocity(np.array(self.config['initial_velocity']))
    

    def extract_franka_mesh_data_combined(self, target_franka):
        """
        Extract and combine all mesh data into single arrays with transformations applied.
        
        Returns:
            vertices: torch tensor of all transformed vertices
            faces: torch tensor of all faces (with proper indexing)
            colors: torch tensor of per-vertex colors
        """
        
        all_vertices = []
        all_faces = []
        all_colors = []
        
        vertex_offset = 0
        sim_vgeoms_render_T = target_franka.solver._vgeoms_render_T
        
        for vgeom in target_franka.vgeoms:
            verts = vgeom.vmesh.verts  # shape: (N, 3)
            faces = vgeom.vmesh.faces
            
            # Get transformation matrix for this vgeom
            cur_render_T = sim_vgeoms_render_T[vgeom.idx][0]  # shape: (4, 4), remove batch dim
            
            # Apply transformation to vertices
            # Convert vertices to homogeneous coordinates (N, 4)
            verts_homogeneous = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
            
            # Apply transformation: (N, 4) @ (4, 4)^T = (N, 4)
            verts_transformed = verts_homogeneous @ cur_render_T.T
            
            # Convert back to 3D coordinates (N, 3)
            verts_transformed = verts_transformed[:, :3]
            
            # Get color from surface
            surface = vgeom.vmesh.surface
            if hasattr(surface, 'diffuse_texture') and surface.diffuse_texture is not None:
                color = surface.diffuse_texture.color
            elif surface.color is not None:
                color = surface.color
            else:
                color = (0.5, 0.5, 0.5)
            
            # Offset faces by current vertex count
            faces_offset = faces + vertex_offset
            
            # Create per-vertex colors
            vertex_colors = np.tile(color, (len(verts), 1))
            
            all_vertices.append(verts_transformed)
            all_faces.append(faces_offset)
            all_colors.append(vertex_colors)
            
            vertex_offset += len(verts)
        
        vertices = torch.from_numpy(np.vstack(all_vertices)).to(self.device, dtype=torch.float32) # + self.franka_pos
        faces = torch.from_numpy(np.vstack(all_faces)).to(self.device, dtype=torch.int32)
        colors = torch.from_numpy(np.vstack(all_colors)).to(self.device, dtype=torch.float32)
        
        return vertices, faces, colors

@register_case('generic')
class GenericCaseHandler(CaseHandler):
    """
    A generic case handler for dynamically generated cases.
    """
    def detect_ground_plane(self, ground_plane):
        """Use background-estimated ground z when available; otherwise fallback to bbox-based plane."""
        self.ground_anchor = self.all_obj_occupied_lower_bound.cpu().numpy().copy()
        auto_ground_z = self.config.get('auto_ground_z', None)

        if auto_ground_z is not None:
            clearance = float(self.config.get('ground_clearance', 0.05))
            self.ground_anchor[2] = float(auto_ground_z - clearance)
        else:
            clearance = float(self.config.get('ground_clearance', 0.05))
            self.ground_anchor[2] = float(self.ground_anchor[2] - clearance)

        self.normal = np.array([0, 0, 1])
        self.scene.add_entity(
            material=gs.materials.Rigid(
                rho=1000.0 if 'plane_rho' not in self.config else self.config['plane_rho'],
                friction=5 if 'plane_friction' not in self.config else self.config['plane_friction'],
                coup_friction=5.0 if 'plane_coup_friction' not in self.config else self.config['plane_coup_friction'],
                coup_softness=0.002 if 'plane_coup_softness' not in self.config else self.config['plane_coup_softness'],
            ),
            morph=gs.morphs.Plane(
                pos=(self.ground_anchor[0], self.ground_anchor[1], self.ground_anchor[2]),
                normal=self.normal,
            ),
        )

    def custom_simulation(self, sid):
        force_direction = np.array(self.config.get('force_direction', [0.0, 0.0, 0.0]))
        force_norm = np.linalg.norm(force_direction)
        if force_norm > 1e-6:
            force_direction = force_direction / force_norm
            force_direction = force_direction.reshape(1, 3)
            force_strength = self.config.get('force_strength', 1.0)
            self.all_objs[0].solver.apply_links_external_force(force=force_direction * force_strength, links_idx=[self.all_objs[0].idx])    
        
        return None

def get_case_handler(case_name: str, config, all_obj_info, device) -> CaseHandler:
    """
    Factory function to return the corresponding CaseHandler instance based on the case name.
    """
    if case_name not in CASE_REGISTRY:
        print(f"Warning: Unknown case name '{case_name}'. Falling back to 'generic' CaseHandler.")
        case_name = 'generic'
        
    # Dynamically get the class object
    CaseClass = CASE_REGISTRY[case_name]
    
    # Instantiate the class object and return
    # Pass all the parameters required by CaseHandler.__init__
    return CaseClass(config, all_obj_info, device)