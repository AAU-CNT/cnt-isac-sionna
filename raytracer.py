#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 08:39:10 2024

This file follows the guide one the sionna website to do raytraced channel generation
https://nvlabs.github.io/sionna/rt/tutorials/Introduction.html

In case that you are running windows and do not have a CUDA-enabled GPU, you
must install LLVM. For mose details see: 
    https://robinw.dk/tutorial/linux/managing-clean-python/#windows

I am assuming that you have installed the following packages:
    spyder
    numpy
    plotly
    pandas
    mpmath
    scipy
    sionna

@author: robin
"""

import os
import platform
import shutil
import csv
import pickle
import time

import gc

import xml.etree.ElementTree as ET
GPU_ON = False

if os.name == 'nt':
    os.environ['DRJIT_LIBLLVM_PATH'] = r"C:\Program Files\LLVM\bin\LLVM-C.dll"
system = platform.system().lower()
if os.name == 'posix' and system == 'linux':
    import platform
    try:
        distro = platform.freedesktop_os_release()
    except Exception:
        distro = {}
    distro_id = distro.get('ID', '').lower()
    version_id = distro.get('VERSION_ID', '').lower()


    id_ver = f"{distro_id} {version_id}".strip()
    if "{} {}".format(distro['ID'], distro['VERSION_ID']) == 'fedora 42':
        os.environ['DRJIT_LIBLLVM_PATH'] = r"/usr/lib64/libLLVM-20.so"
        print('importing!')

    if "{} {}".format(distro['ID'], distro['VERSION_ID']) == 'ubuntu 22.04':
        os.environ['DRJIT_LIBLLVM_PATH'] = r"/usr/lib/llvm-20/lib/libLLVM.so"
        print('importing!')
    if "{} {}".format(distro['ID'], distro['VERSION_ID']) == 'ubuntu 24.04':
        os.environ['DRJIT_LIBLLVM_PATH'] = r"/usr/lib/llvm-18/lib/libLLVM.so.1"
        print('importing!')

    if "{} {}".format(distro['ID'], distro['VERSION_ID']) == 'linuxmint 22.1':
        os.environ['DRJIT_LIBLLVM_PATH'] = r"/usr/lib/llvm-18/lib/libLLVM.so"
        print('importing!')

    # FOR THE SERVER! WHEN GPU ENABLED, you should pass an int not a drjit.uint...
    if distro['ID'] == 'ubuntu':
        GPU_ON = True

elif system == 'darwin':  # macOS
        # Try Homebrew first (works for both /opt/homebrew and /usr/local)
        brew = shutil.which("brew")
        candidates = []
        if brew:
            # Ask brew where llvm lives
            try:
                import subprocess
                prefix = subprocess.check_output([brew, "--prefix", "llvm"], text=True).strip()
                candidates.extend([
                    f"{prefix}/lib/libLLVM.dylib",
                    f"{prefix}/lib/libLLVM-20.dylib",
                ])
            except Exception:
                print("failed 1")
                pass
        
        # Add typical Homebrew prefixes (Apple Silicon & Intel)
        candidates.extend([
            "/opt/homebrew/opt/llvm/lib/libLLVM.dylib",
            "/usr/local/opt/llvm/lib/libLLVM.dylib",
            "/opt/homebrew/opt/llvm/lib/libLLVM-20.dylib",
            "/usr/local/opt/llvm/lib/libLLVM-20.dylib",
        ])

        for candidate in candidates:
            if os.path.exists(candidate):
                os.environ['DRJIT_LIBLLVM_PATH'] = candidate
                print('importing!')
                break
import mitsuba as mi
import drjit as dr
#mi.set_variant("cuda_ad_rgb")
import sionna as si
si.__version__='1.2.1'
if si.__version__ == '0.19.2':
    si.config.seed = 41

import tensorflow as tf
import numpy as np


#import matplotlib
#matplotlib.rcParams['mathtext.fontset'] = 'cm'
#matplotlib.rcParams['font.family'] = 'STIXGeneral'






class Channel:
    
    def __init__(self, 
                 max_n_rx = 100, 
                 nTx = None, 
                 txPos = tf.constant([[8.5, 21.0, 27.0]], dtype=tf.float32), 
                 txName = None, 
                 nRx = None, 
                 rxPos = tf.constant([[45, 90, 1.5]], dtype=tf.float32), 
                 rxName = None, 
                 frequency = 2.14e9,
                 scenePath = 'munich_small.xml',
                 skip_scene_creation=False):
        
        width = 1655
        height = 1150
        
        #self.resolution = [480*3,320*3] # increase for higher quality of renderings
        self.path_seed = 0
        self.resolution = (width, height)
        self.num_subcarriers = 8001
        self.subcarrier_spacing = 1e6
        self.rxPos = rxPos
        self.valid_rx_pos = tf.ones_like(rxPos, dtype=tf.bool)

        
        self.nTx = nTx
        self.txPos = txPos
        self.txName = txName
        self.nRx = nRx
        self.rxPos = rxPos
        self.rxName = rxName
        self.frequency = frequency
        
        self.max_n_rx = max_n_rx
        
        if nTx is None:
            self.nTx = tf.ones(txPos.shape[0])
        if txName is None:
            self.txName = ['tx' + str(n) for n in range(txPos.shape[0])]
        if nRx is None:
            self.nRx = tf.ones(rxPos.shape[0])
        if rxName is None:
            self.rxName = ['rx' + str(n) for n in range(rxPos.shape[0])]

        n_rx_terminals = rxPos.shape[0] if (rxPos.shape[0] < self.max_n_rx) else self.max_n_rx
        rxPos_in = self.rxPos[0:n_rx_terminals,:]
        nRx_in = self.nRx[0:n_rx_terminals]
        rxName_in = self.rxName[0:n_rx_terminals]
        
        if not skip_scene_creation:
            self.scene = self.createScene(self.nTx, self.txPos, self.txName, nRx_in, rxPos_in, rxName_in, self.frequency, scenePath)
        
        self.unique_materials_created = False
        
        self.scene_object_positions = None
        
    def createScene(self, nTx, txPos, 
                    txName, nRx, rxPos, 
                    rxName, frequency, scenePath):
        
        # Loads the munich scene.
        if si.__version__ == '0.19.2':
            scene = si.rt.load_scene(filename=scenePath)
        else:
            scene = si.rt.load_scene(filename=scenePath, merge_shapes = False)
        
        # Configure antenna array for all transmitters
        scene.tx_array = si.rt.PlanarArray(num_rows=1,
                                     num_cols=int(nTx[0]),
                                     vertical_spacing=0.5,
                                     horizontal_spacing=0.5,
                                     pattern="dipole",
                                     polarization="V")
        
        # Configure antenna array for all receivers
        scene.rx_array = si.rt.PlanarArray(num_rows=1,
                                     num_cols=int(nRx[0]),
                                     vertical_spacing=0.5,
                                     horizontal_spacing=0.5,
                                     pattern="dipole",
                                     polarization="V")
        
        # OBSERVE THAT IN PUTTING THE POLARIZAITON TO SOMETHING DUAL-POLIARZED,
        # YOU DOUBLE THE NUMBER OF ANTENNAS IN THE SIMLATION.
        
        # Create transmitter
        if si.__version__ == '0.19.2':
            for txIndx in range(txPos.shape[0]):
                tx = si.rt.Transmitter(name=txName[txIndx],
                                 position=txPos[txIndx,:])
                
                # Add transmitter instance to scene
                scene.add(tx)
            
            # Create a receiver
            for rxIndx in range(rxPos.shape[0]):
                rx = si.rt.Receiver(name=rxName[rxIndx],
                              position=rxPos[rxIndx,:],
                              orientation=[0,0,0])
                
                # Add receiver instance to scene
                scene.add(rx)
        else:
            for txIndx in range(txPos.shape[0]):
                tx = si.rt.Transmitter(name=txName[txIndx],
                                 position=txPos[txIndx,:],
                                 display_radius=2)
                
                # Add transmitter instance to scene
                scene.add(tx)
            
            # Create a receiver
            for rxIndx in range(rxPos.shape[0]):
                rx = si.rt.Receiver(name=rxName[rxIndx],
                              position=rxPos[rxIndx,:],
                              orientation=[0,0,0],
                              display_radius=2)
                # Add receiver instance to scene
                scene.add(rx)

        scene.frequency = frequency # in Hz; implicitly updates RadioMaterials

        scene.synthetic_array = False # If set to False, ray tracing will be done per antenna element (slower for large arrays)


        txPos[0,:]

        if si.__version__ == '0.19.2':
            my_cam = si.rt.Camera("my_cam", position=[45-0.1,90,400], 
                                  look_at=[45,90,28])
            
            if 0:
                height = 220
                x_shift = 10
                y_shift = 5
                
                my_cam = si.rt.Camera("my_cam", position=txPos[0,:] + [-x_shift,y_shift-0.1,height], 
                                      look_at=txPos[0,:] + [-x_shift,y_shift,0])
                
            scene.add(my_cam)

        return scene
    
    def add_material_noise(self, noise_std = 1):
        
        scene = self.scene
        rng = np.random.default_rng()
        
        position_limit = 2
        
        #noise = lambda : rng.normal(loc = 0, scale = noise_std)
        noise = lambda : rng.uniform(low = 1.5, high = 30)
        position_noise = lambda : np.array( [rng.uniform(low = -position_limit, high = position_limit), 
                                             rng.uniform(low = -position_limit, high = position_limit), 0] )
        
        read_positions = True
        if not (self.scene_object_positions is None):
            if (self.scene_object_positions.shape[0] == len(scene.objects)):
                read_positions = False
        
        if read_positions:
            self.scene_object_positions = np.zeros( (len(scene.objects), 3) )
            for indx, (_, scene_object) in enumerate(scene.objects.items()):
                self.scene_object_positions[indx,:] = scene_object.position.numpy()
        
        if not self.unique_materials_created:
            self.unique_materials_created = True
            for name, scene_object in scene.objects.items():
                
                if si.__version__ == '0.19.2':
                    custom_material = si.rt.RadioMaterial('{}_mat'.format(name),
                                        relative_permittivity=scene_object.radio_material.relative_permittivity.numpy(),
                                        conductivity=scene_object.radio_material.conductivity.numpy(),
                                        scattering_coefficient=scene_object.radio_material.scattering_coefficient.numpy(),
                                        xpd_coefficient=scene_object.radio_material.xpd_coefficient.numpy(),
                                        scattering_pattern=si.rt.LambertianPattern())
                    
                else:
                    custom_material = si.rt.RadioMaterial('{}_mat'.format(name),
                                        relative_permittivity=scene_object.radio_material.relative_permittivity.numpy(),
                                        conductivity=scene_object.radio_material.conductivity.numpy(),
                                        scattering_coefficient=scene_object.radio_material.scattering_coefficient.numpy(),
                                        xpd_coefficient=scene_object.radio_material.xpd_coefficient.numpy(),
                                        scattering_pattern='lambertian',
                                        color=tuple(rng.uniform(0, 1, 3)))
                
                scene_object.radio_material = custom_material

        
        # Move objects
        for indx, (_, scene_object) in enumerate(scene.objects.items()):
            scene_object.position = self.scene_object_positions[indx,:] + position_noise()
        
        # Randomize materials
        for key in scene.radio_materials:
            rel_perm = noise()
            if rel_perm < 1:
                rel_perm = 1
                
            scene.radio_materials[key].relative_permittivity = rel_perm
                
    def _move_receivers(self, batch_lim = None, batch=None, specfic_locations=None):
        scene = self.scene
        
        # Remove old receivers
        for name in scene.receivers:
            scene.remove(name)
        
        if not (specfic_locations is None):
            for rx_n in range(specfic_locations.shape[0]):
                    rx = si.rt.Receiver(name='rx' + str(rx_n),
                                  position=specfic_locations[rx_n,:],
                                  orientation=[0,0,0])
                    scene.add(rx)
        else:
            if batch is None:
                if batch_lim is None:
                    batch_lim =  [0, self.rxPos.shape[0]]
                batch = np.arange(batch_lim[0], batch_lim[1])
            # Add new receivers
            for rx_n in batch:
                    print("rx_n:",rx_n)
                    rx = si.rt.Receiver(name=self.rxName[rx_n],
                                  position=self.rxPos[rx_n,:],
                                  orientation=[0,0,0])
                    scene.add(rx)
                    
    
    def _computePaths(self, seed = 42):
           
        # Compute propagation paths
        # Number of rays shot into directions defined
        # by a Fibonacci sphere , too few rays can
        # lead to missing paths
        
        scene = self.scene
        
        if si.__version__ == '0.19.2':
            #paths = scene.compute_paths(max_depth=6,
            #               num_samples=int(1e6),
            #               los=True,
            #               reflection=True,
            #               diffraction=True,
            #               scattering=True,
            #               edge_diffraction=True,
            #               ris=False,
            #               )
            paths=scene.compute_paths(max_depth=6,
                           num_samples=int(1e4),
                           los=True,
                           reflection=True,
                           diffraction=True,
                           scattering=True,
                           edge_diffraction=True,
                           ris=False,
                           )
        else:
            solver = si.rt.PathSolver()
            paths = solver(scene=scene, 
                       max_depth=6,#number of bounces
                       samples_per_src=int(1e4),
                       specular_reflection=True,
                       diffuse_reflection=True,
                       refraction=True,
                       ) 
        return paths
        
    def illustrate(self, paths = None, remove_receivers = False, 
                   receiver_ids = None):
        
        scene = self.scene
        
        if si.__version__ == '0.19.2':
            if (receiver_ids is None):
                if remove_receivers:
                    self._move_receivers([0,0])
                else:
                    self._move_receivers()
            else:
                self._move_receivers(batch=receiver_ids)
                
                
            if paths is None:
                scene.render("my_cam", show_devices=True, 
                             resolution=self.resolution);
            else:
                scene.render("my_cam", paths=paths, show_devices=True, 
                             show_paths=True, resolution=self.resolution);
            
        else:
            cam = si.rt.Camera(position=[45-0.1,90,800], look_at=[45,90,28])

            if paths is None:
                scene.render(camera=cam, show_devices=True, 
                             resolution=self.resolution);
            else:
                scene.render(camera=cam, paths=paths, show_devices=True, 
                             resolution=self.resolution);
                
    def prune_scene_objects(self, xlim = [-200, 200], ylim = [-250, 250]):
        
        scene = self.scene
        rem = []
        
        for name, scene_object in scene.objects.items():
            
            delete = ((scene_object.position[0] < xlim[0]) or 
                      (scene_object.position[0] > xlim[1]) or 
                      (scene_object.position[1] < ylim[0]) or
                      (scene_object.position[1] > ylim[1]))
            if delete:
                if si.__version__ == '0.19.2':
                    #scene_object.position = [0,0, -1000]
                    pass
                    #scene.remove(name)
                rem.append(name)
        
        if si.__version__ == '0.19.2':
            print("sionna v. 0.19.2 does not support removal of scene objects, so we just skip this step.")
        else:
            scene.edit(remove=rem)

    def prep_scene (self, receivers):
        for name in list(self.scene.receivers):
            self.scene.remove(name)

        self.names = [str(i) for i in range(int(receivers))]
        for i in range(int(receivers)):
            rx = si.rt.Receiver(
                name=self.names[i],
                position=tf.zeros(3),     # keep as TF tensor
                orientation=[0.0, 0.0, 0.0]
            )
            self.scene.add(rx)
    

    def get_cir(self, specific_receiver=None, run_in_batches=False):
        a_s = list()
        tau_s = list()
            
        if specific_receiver is not None:
            receiver_pos = _to_numpy(specific_receiver)
        else:
            receiver_pos = self.rxPos
        n_receivers = receiver_pos.shape[0]
    
        if run_in_batches:
            blocks = int(n_receivers / self.max_n_rx)
            if n_receivers % self.max_n_rx:
                blocks += 1
                        
            for batch in range(blocks):
                
                print('Running batch {} out of {}'.format(batch + 1, blocks))
                
                if batch == blocks-1:
                    batch_lim = [batch*self.max_n_rx, n_receivers]
                else:
                    batch_lim = [batch*self.max_n_rx, (batch+1)*self.max_n_rx]
                
                self._move_receivers(specfic_locations=receiver_pos[batch_lim[0]:batch_lim[1],:])
                
                paths = self._computePaths()
                
                a, tau = paths.cir(out_type='tf')

                if not (a.shape[-1] == 1):
                    Exception('Time_steps not supported')
                
                if si.__version__ == '0.19.2':
                    for n in range(a.shape[1]):
                        a_s.append( a[:, n, :, :, :, :] )
                        tau_s.append( tau[:, n, :, :, :] )
                else:
                    for n in range(a.shape[0]):
                        a_s.append( a[n, :, :, :, :, :] )
                        tau_s.append( tau[n, :, :] )
        else:
            # ensure TF tensor
            #specific_receiver = tf.convert_to_tensor(specific_receiver, dtype=tf.float32)
            _set_receiver_positions(self.scene, self.names,
                                    receiver_pos[:len(self.names)])
            
            #for i in range(len(self.names)):
            #    rx = self.scene.get(self.names[i])
            #    rx.position = receiver_pos[i]
            paths = self._computePaths()
            a, tau = paths.cir(out_type='tf')  # tf.Tensor, tf.Tensor
        
            if not (a.shape[-1] == 1):
                Exception('Time_steps not supported')
            
            if si.__version__ == '0.19.2':
                for n in range(a.shape[1]):
                    a_s.append( a[:, n, :, :, :, :, :] )
                    tau_s.append( tau[:, n, :, :, :, :] )
            else:
                a_s   = tf.identity(a)
                tau_s = tf.identity(tau)

        return a_s, tau_s

def _to_numpy(x):
    #if isinstance(x, torch.Tensor):
    #    return x.detach().cpu().numpy().astype(np.float32)
    #if hasattr(x, 'numpy'):       # TF tensor fallback (shouldn't appear)
    #    return x.numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)

def _set_receiver_positions(scene, names, positions):
    """Update receiver positions from an [N, 3] array via one opaque buffer.

    Assigning plain Python floats (e.g. ``rx.position = [x, y, z]``) bakes
    the coordinates into the traced kernels as literals, which changes the
    kernel hash and forces a full JIT recompile of the path solver on every
    call (~2 s at 512 receivers). Routing the values through a single opaque
    Dr.Jit buffer keeps the kernel structure stable so it is compiled once
    and reused.
    """
    positions = _to_numpy(positions)
    buf = mi.Point3f(positions[:, 0], positions[:, 1], positions[:, 2])
    dr.make_opaque(buf)
    for i, name in enumerate(names):
        scene.get(name).position = dr.gather(mi.Point3f, buf, i)


def power_quantile(h_in, eps=None):
    h = tf.math.square(tf.abs(h_in))  # |h|^2

    def quantile_fn(eps_local):
        eps_t = tf.convert_to_tensor(eps_local, dtype=h.dtype)
        M = tf.shape(h)[1]
        k = tf.cast(tf.floor(eps_t * tf.cast(M - 1, h.dtype)), tf.int32)

        h_sorted = tf.sort(h, axis=1)
        # k is scalar → gather along axis=1 with batch_dims=0
        h_q = tf.gather(h_sorted, k, axis=1)  # shape [B]
        return h_q

    return quantile_fn if eps is None else quantile_fn(eps)
        

def log_to_power(x):
    return 10**(x/10)

def power_to_log(x):
    # 10 * log10(x) = 10 * log(x) / log(10)
    return 10.0 * tf.math.log(x) / tf.math.log(tf.constant(10.0, dtype=x.dtype))

def create_channel_munich_tf(txPos=None):
    xlim=[-90,194]#[-400,400]
    ylim = [-118,185]#[-400,400]
    step_size = 1
    # Base receiver offset
    rxPos_base = tf.constant([[0.0, 0.0, 1.15]])

    # Grid in TF (half-open ranges like np.mgrid start:stop:step)
    x = tf.range(xlim[0],  xlim[1],  delta=step_size, dtype=tf.float32)
    y = tf.range(ylim[0],  ylim[1],  delta=step_size, dtype=tf.float32)
    z = tf.constant([0.0], dtype=tf.float32)

    # Mesh + flatten to [N, 3]
    X, Y, Z = tf.meshgrid(x, y, z, indexing='ij')                # [Nx, Ny, Nz]
    XYZ     = tf.stack([X, Y, Z], axis=-1)                       # [Nx, Ny, Nz, 3]
    rxPos_grid = tf.reshape(XYZ, [-1, 3]) + rxPos_base           # [N, 3]
    nRx = tf.shape(rxPos_grid)[0]
    
    if txPos is None:
        txPos= np.array([[ -64, -69,  1.15],[ 50, 120,  1.15],[ 135.5, 173.2,  1.15],[148, -28.27,  1.15],[0, -102,  1.15]])
        #txPos= np.array([[ -64, -69,  1.15],[ 50, 120,  1.15],[ 135.5, 173.2,  1.15],[148, -28.27,  1.15],[0, -102,  1.15], 
        #                 [76, 11,  1.15],[0, 150,  1.15],[-20, 60,  1.15],[32, 41.5,  1.15],[76, -119,  1.15],])
        #txPos= np.array([[ -64, -69,  1.15],[ 50, 120,  1.15],[ 135.5, 173.2,  1.15],[148, -28.27,  1.15],[0, -102,  1.15], 
        #                 [76, 11,  1.15],[0, 150,  1.15],[-20, 60,  1.15],[32, 41.5,  1.15],[76, -119,  1.15],
        #                 [112,100, 1.15],[67,-45, 1.15],[114,-76, 1.15],[-37,31.6, 1.15],[95,179, 1.15]])
        #txPos= np.array([[ -64, -69,  1.15],[ 50, 120,  1.15],[ 135.5, 173.2,  1.15],[148, -28.27,  1.15],[0, -102,  1.15], 
        #                 [76, 11,  1.15],[0, 150,  1.15],[-20, 60,  1.15],[32, 41.5,  1.15],[76, -119,  1.15],
        #                 [112,100, 1.15],[67,-45, 1.15],[114,-76, 1.15],[-37,31.6, 1.15],[95,179, 1.15],
        #                 [-60,170, 1.15],[-49,-18, 1.15],[17,89, 1.15],[74,74, 1.15],[-40,-86, 1.15]])


    return rxPos_grid, txPos, nRx, xlim, ylim

def make_pruned_scene(scenePath, xlim, ylim):
    """Write (and cache) a copy of the scene XML with every shape whose
    centre lies outside xlim/ylim removed.

    Pruning needs per-object positions, which requires an unmerged scene —
    but unmerged scenes are slow to trace. Baking the pruning into the XML
    once lets the pruned scene be loaded with merge_shapes=True afterwards,
    giving identical geometry/materials at merged-scene speed.

    Returns the path of the pruned XML file.
    """
    base, ext = os.path.splitext(scenePath)
    tag = f"x{xlim[0]}_{xlim[1]}_y{ylim[0]}_{ylim[1]}".replace('-', 'm')
    out_path = f"{base}_pruned_{tag}{ext}"
    if (os.path.exists(out_path)
            and os.path.getmtime(out_path) >= os.path.getmtime(scenePath)):
        return out_path

    print(f"Pruning scene XML -> {out_path}")
    scene = si.rt.load_scene(filename=scenePath, merge_shapes=False)
    remove = set()
    for name, obj in scene.objects.items():
        pos = np.array(obj.position)
        if (pos[0] < xlim[0] or pos[0] > xlim[1] or
                pos[1] < ylim[0] or pos[1] > ylim[1]):
            remove.add(name)
    # Drop the unmerged scene before any further tracing happens: its
    # objects would otherwise stay registered with Dr.Jit and slow down
    # kernels traced for the scene that is actually used.
    del scene
    gc.collect()

    tree = ET.parse(scenePath)
    root = tree.getroot()
    n_removed = 0
    for shape in list(root.findall("shape")):
        sid = shape.attrib.get("id", "")
        name = sid[5:] if sid.startswith("mesh-") else sid
        if sid in remove or name in remove:
            root.remove(shape)
            n_removed += 1
    print(f"Pruned {n_removed} shapes, {len(remove)} objects outside limits")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def create_channel_munich(txPos=None):
    rxPos_grid, txPos, nRx, xlim, ylim = create_channel_munich_tf(txPos)

    # Choose a batch size for receivers (GPU_ON is your flag)
    max_n_rx = 512
    if si.__version__ == '0.19.2':
        max_n_rx = 20 if GPU_ON else 100
        max_n_rx = 50

    # Boundary conversion (only here!)
    #rxPos_np = rxPos_grid.numpy()
    #txPos_np = txPos.numpy()

    print("Creating scene")


    ################
    #NEW BIG CHANGE
    ################
    scene_file = make_pruned_scene("scenario/munich_small/munich_small.xml",
                                   xlim=xlim, ylim=ylim)

    ch = Channel(max_n_rx=max_n_rx,
                 txPos=txPos,
                 rxPos=rxPos_grid,
                 frequency=6e9,
                 scenePath=scene_file)
                 #scenePath="scenario/munich_small/munich_small.xml")
    
    ch.prune_scene_objects(
        xlim=[xlim[0], xlim[1]],
        ylim=[ylim[0], ylim[1]]
    )
    include_a_car=False
    if include_a_car:
        from sionna.rt.scene import low_poly_car
        def create_car():
            mat=si.rt.ITURadioMaterial("car_mat","metal", thickness=0.01)
            car=[si.rt.SceneObject(fname=low_poly_car, name=f"cae-{i}", radio_material=mat) for i in range(100)]
            ch.scene.edit(add=car)
            for i in range(100):
                car[i].position=(int(i/2),int(i/2),1)
            
            return car
        create_car()
        #ch.scene.render()

    brick_fraction = 0.0  # fraction of scene objects to turn into brick
    brick_seed = 42       # RNG seed so the selection is reproducible
    if brick_fraction > 0.0:
        def make_some_brick():
            # One shared brick material; EM properties auto-update from the
            # scene frequency (ITU-R P.2040). thickness in metres.
            brick = si.rt.ITURadioMaterial("brick_some", "brick", thickness=0.2)
            names = list(ch.scene.objects.keys())
            rng = np.random.default_rng(brick_seed)
            n_brick = int(round(brick_fraction * len(names)))
            brick_names = rng.choice(names, size=n_brick, replace=False)
            for name in brick_names:
                ch.scene.objects[name].radio_material = brick
            return brick
        make_some_brick()

    return ch, rxPos_grid, txPos



def create_channel_AAU():
    
    # Compute receiver positions
    rxPos_base = np.array([[60, -0.90699, 1.15]])
    xlim = 80
    ylim = 50
    step_size = 4
    XYZ = np.mgrid[-xlim:xlim:step_size, -ylim:ylim:step_size, 0:1].reshape(3,-1).transpose()
    rxPos_grid = XYZ + rxPos_base
    #rxPos_grid = rxPos_grid[0:2,:]
    nRx = rxPos_grid.shape[0]
    
    txPos= np.array([[70, -0.9, 1.15]])
    
    print('Creating scene')
    max_n_rx = nRx
    if si.__version__ == '0.19.2':
        max_n_rx = 20 if GPU_ON else 100
        max_n_rx = 50
    ch = Channel(max_n_rx=max_n_rx, txPos=txPos, 
                 rxPos=rxPos_grid, frequency=6e9,
                 scenePath='scenario/FBV7/FBV7.xml')
    
    if not si.__version__ == '0.19.2':
        print('Pruning scene')
        ch.prune_scene_objects(xlim=[-200, 200] + rxPos_base[0][0], ylim=[-250, 250] + rxPos_base[0][1])
    
    return ch, rxPos_grid, txPos


def get_realization(ch, individual_sensor=None, scenario='qorvo', seed=0, save_and_load=False, run_in_batches=False):
    if not save_and_load:
        #print('Loading skipped, simulating realization')
        a, tau = ch.get_cir(individual_sensor, run_in_batches)
    else:
        try:
            fn = 'results_{}/RT.pkl'.format(scenario)
            with open(fn, "rb") as f:
                contents = pickle.load(f)  # Load and unpack the tuple
            a = contents[0]
            tau = contents[1]
        except:
            print('Loading failed, simulating realization')
            a, tau = ch.get_cir(run_in_batches=run_in_batches)
            
            with open(fn, "wb") as f:
                pickle.dump((a, tau), f)
                
    return a, tau

        

def compute_save_mask(a, threshold):
    users = len(a)
    save_mask = np.ones(users, dtype=bool)
    
    for n in range(users):
        if np.all(np.isclose(abs(a[n]), 0.0)):
        #if np.all(np.abs(a[n]) < threshold):
            save_mask[n] = False
    return save_mask

def compute_pos_save_mask(pos_grid, ref_grid,exclusion_rules=None):
    matches = np.all(np.isclose(pos_grid[:, None, :], ref_grid[None, :, :],), axis=-1)
    mask = np.any(matches, axis=1).copy()   # copy because we will modify it

    # 2. Apply your new x-y (or x-y-z) exclusion rules
    if exclusion_rules is not None:
        if isinstance(exclusion_rules, dict):
            exclusion_rules = [exclusion_rules]   # allow single dict for convenience

        x = pos_grid[:, 0]
        y = pos_grid[:, 1]
        z = pos_grid[:, 2]

        for rule in exclusion_rules:
            exclude = np.ones(len(pos_grid), dtype=bool)
            if 'x_min_lim' in rule:
                exclude &= (x > rule['x_min_lim'])
            if 'x_max_lim' in rule:
                exclude &= (x < rule['x_max_lim'])
            if 'y_min' in rule:
                exclude &= (y >= rule['y_min'])
            if 'y_max' in rule:
                exclude &= (y <= rule['y_max'])
            mask &= ~exclude

    return mask
def create_channel_qorvo(center_frequency, skip_scene_creation=False, txPos=None):
    # Compute receiver positions
    rxPos_base = np.array([[60, -0.90699, 1.15]])
    xlim = 80
    ylim = 50
    step_size = 1
    XYZ = np.mgrid[-xlim:xlim:step_size, -ylim:ylim:step_size, 0:1].reshape(3,-1).transpose()
    rxPos_grid = XYZ + rxPos_base
    #rxPos_grid = rxPos_grid[0:2,:]
    nRx = rxPos_grid.shape[0]
    if txPos is None:
        #txPos= np.array([[70, -0.9, 1.15], [41.5, 10, 1.15], [85, 37, 1.15], [53, -10.7, 1.15], [56, 44.5, 1.15]])
        #txPos= np.array([[ 60,   0,  1.15],[ 80,  36,  1.15],[120,  14,  15],[115, -40,  15],[-15 ,-20,  15]])
        txPos= np.array([[-18, -20, 1.15], [60, 0, 1.15], [80, 35, 1.15], [120, 15, 1.15], [115, -40, 1.15],[50, -30, 1.15], [-5, 45, 1.15], [40, 20, 1.15]])
    
    
    print('Creating scene')
    max_n_rx = 200
    if si.__version__ == '0.19.2':
        max_n_rx = 20 if GPU_ON else 100
        max_n_rx = 50
        
    ch = Channel(max_n_rx=max_n_rx, 
                 txPos=tf.convert_to_tensor(txPos, dtype=tf.float32), 
                 rxPos=tf.convert_to_tensor(rxPos_grid, dtype=tf.float32), 
                 frequency=center_frequency,
                 scenePath='scenario/FBV7/FBV7.xml',
                 skip_scene_creation=skip_scene_creation)
    
    return ch, rxPos_grid, txPos

def deembed_carrier_frequency(a, tau, freq):
    for n in range(len(a)):
        a[n] = a[n] * np.exp(1j * tau[n] * freq)
    return a

class bandlimiter:
    def __init__(self, bandwidth):
        # Use float32 / complex64 everywhere
        self.eps = tf.constant(1e-6, dtype=tf.float32)      # threshold for sinc near 0 (float32-safe)
        self.pi  = tf.constant(np.pi, dtype=tf.float32)
        self.two_pi = tf.constant(2*np.pi, dtype=tf.float32)

        if isinstance(bandwidth, (float, int, np.floating, np.integer)):
            self.bw = tf.constant(float(bandwidth), dtype=tf.float32)
            self.shift = tf.constant(0.0, dtype=tf.float32)
        else:
            if len(bandwidth) > 1:
                bw = float(bandwidth[1] - bandwidth[0])
                shift = float((bandwidth[1] + bandwidth[0]) / 2.0)
                self.bw = tf.constant(bw, dtype=tf.float32)
                self.shift = tf.constant(shift, dtype=tf.float32)
            else:
                raise Exception("unhandled bandwidth specification, use either scalar or vector of 2 entries")

    def sinc(self, x):
        # x: float32 -> returns complex64
        x = tf.cast(x, tf.float32)
        pix = x * self.pi
        y = tf.math.sin(pix) / pix
        y = tf.where(tf.abs(x) < self.eps, tf.ones_like(y, dtype=tf.float32), y)
        return tf.cast(y, tf.complex64)

    def filt(self, x, shift):
        # x: float32, shift: float32 -> complex64 filter
        x = tf.cast(x, tf.float32)
        shift = tf.cast(shift, tf.float32)

        phase = self.two_pi * shift * x                      # float32
        phase_c = tf.complex(tf.zeros_like(phase), phase)    # complex64 = j*phase

        return self.sinc(x * self.bw) * tf.cast(self.bw, tf.complex64) * tf.exp(phase_c)

    def bandlimit_cir(self, time_axis, a, tau, batched=True):
        # time_axis: (T)
        time_axis = tf.cast(time_axis, tf.float32)

        a = tf.squeeze(a)
        tau = tf.squeeze(tau)

        if not batched:
            a   = a[tf.newaxis, ...]    # (1, U, Tx, P)
            tau = tau[tf.newaxis, ...]  # (1, U, Tx, P)

        tau32 = tf.cast(tau, tf.float32)       # real for sinc argument
        a_c   = tf.cast(a, tf.complex64)       # complex amplitudes

        tau_exp = tau32[..., tf.newaxis]       # (B, U, Tx, P, 1)
        t_exp   = time_axis[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :]  # (1,1,1,1,T)
        x = t_exp - tau_exp                    # float32

        filt_vals = self.filt(x, self.shift)   # complex64, (B,U,Tx,P,T)
        a_exp = a_c[..., tf.newaxis]           # complex64, (B,U,Tx,P,1)

        cir_B_U_Tx_T = tf.reduce_sum(filt_vals * a_exp, axis=3)  # sum over P -> (B,U,Tx,T)
        cir = tf.squeeze(cir_B_U_Tx_T)

        if not batched:
            cir = tf.squeeze(cir, axis=0)      # (U,Tx,T) or (Tx,T,U) depending on your upstream shapes

        return cir


class bandlimiter_big:
    def __init__(self, bandwidth):
        self.eps = tf.constant(1e-20, dtype=tf.float64)
        self.pi  = tf.constant(np.pi, dtype=tf.float64)
        self.two_pi = tf.constant(2*np.pi, dtype=tf.float64)
        if type(bandwidth) is float:
            self.bw = bandwidth
            self.shift = 0
        elif len(bandwidth) > 1:
            self.bw = bandwidth[1] - bandwidth[0]
            self.shift = (bandwidth[1] + bandwidth[0]) / 2
        else:
            Exception("unhandled bandwidth specification, use either scalar or vector of 2 entries")
    def sinc(self,x):
            x = tf.cast(x, tf.float64)
            pix = x * self.pi
            y=tf.sin(pix) / (pix)
            y = tf.where(tf.abs(x) < self.eps, tf.ones_like(y, tf.float64), y)
            return tf.cast(y, tf.complex128)
        #phase_shift=tf.math.exp(1j * 2 * pi * shift) #maybe tf.math.exp(1j * 2 * pi * shift * x)
    def filt(self,x, shift):
        #return sinc(x * bw) * bw * tf.math.exp(1j * 2 * pi * shift) #maybe tf.math.exp(1j * 2 * pi * shift * x)
        phase = self.two_pi * tf.cast(shift, tf.float64) * x          # float64
        phase_c = tf.complex(tf.zeros_like(phase), phase)        # complex128 = j*phase

        return self.sinc(x * self.bw) * tf.cast(self.bw, tf.complex128) * tf.exp(phase_c)
    def bandlimit_cir(self, time_axis, a, tau, batched=True):
        time_axis = tf.cast(time_axis, tf.float64)
        a=tf.squeeze(a)
        tau=tf.squeeze(tau)
        if not batched:
            a   = a[tf.newaxis, ...]   # (1, U, Tx, P)
            tau = tau[tf.newaxis, ...] # (1, U, Tx, P)
        tau64 = tf.cast(tau, tf.float64)      # real, for sinc input
        a_c   = tf.cast(a, tf.complex128)     # complex, like your loop
        tau_exp = tau64[..., tf.newaxis]
        t_exp   = time_axis[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :]
        x = t_exp - tau_exp
        filt_vals = self.filt(x, self.shift)
        a_exp = a_c[..., tf.newaxis]         # (B, U, Tx, P, 1)
        cir_B_U_Tx_T = tf.reduce_sum(filt_vals * a_exp, axis=3)
        cir=tf.squeeze(cir_B_U_Tx_T)
        if not batched:
            cir = tf.squeeze(cir, axis=0)     # (Tx, T, U)
        return cir


    

if __name__=='__main__':
    with tf.device("/CPU:0"):
        scenario = 'qorvo' # 'munich' | 'AAU'

        # Default settings for bandwidth etc.
        bandwidth=100e6
        time_axis = np.arange(start=0, stop=2e-6, step=1/bandwidth)
        
        if scenario == 'AAU':
            ch, rxPos_grid, txPos = create_channel_AAU()
        elif scenario == 'munich':
            ch, rxPos_grid, txPos = create_channel_munich()
        elif scenario == 'qorvo':
            
            # Obtained from DW1000 User Manual section on CIR and Table 8 of the DW3000 user manual.
            num_samples=1016
            fundamental_frequency = 499.2 * 1e6
            center_frequency = 6489.6 * 1e6 
            time_step = 1 / (fundamental_frequency * 2)
            time_axis = np.arange(0, num_samples) * time_step
            freq_axis = np.fft.fftfreq(num_samples, time_step) + center_frequency
            bandwidth = 499.2 * 1e6
            
            ch, rxPos_grid, txPos = create_channel_qorvo(center_frequency, skip_scene_creation=False)
            
        else:
            Exception('unknown scene')
        
        print('scene created, loading realization')
        
        a_all, tau_all = get_realization(ch, save_and_load=False, run_in_batches=True)      
        print('Computing save_mask and removing blocked sensors')
        # Only save data for points which actually receive something
        use_positional_save_mask=False
        if use_positional_save_mask:
            rules = [
                {"x_min_lim": 123.0, "y_min": -15.0, "y_max": 105.0},
                {"x_max_lim": -50.0, "y_min": 30.0, "y_max": 200.0},
            ]
            ref_grid=np.loadtxt("rxPos_grid_munich.csv", delimiter=",")
            save_mask = compute_pos_save_mask(rxPos_grid, ref_grid, exclusion_rules=rules)
        else:
            save_mask = compute_save_mask(a_all, threshold=1e-4)
        a = list() 
        tau = list() 
        cir = list()
        #for save_entry in np.where(save_mask)[0]:
        print('Computing channel impulse response')
        blcir=bandlimiter(bandwidth)
        for save_entry in np.where(save_mask)[0]:
            cir_impulse = blcir.bandlimit_cir(time_axis=time_axis, a=a_all[save_entry], tau=tau_all[save_entry])
            cir.append(cir_impulse)
            a.append(a_all[save_entry])
            tau.append(tau_all[save_entry])
        try:
            rxPos_grid = rxPos_grid[save_mask, :]
        except:
            save_mask_tf = tf.convert_to_tensor(save_mask, dtype=tf.bool)   # shape [N]
            rxPos_grid = tf.boolean_mask(rxPos_grid, save_mask_tf, axis=0)
        np.savetxt("rxPos_grid.csv", rxPos_grid, delimiter=",")
        cir  = tf.stack(cir, axis=0)
        #a    = tf.stack(a, axis=0)
        #tau  = tf.stack(tau, axis=0)
        
        #cir = bandlimit_cir(bandwidth=bandwidth, time_axis=time_axis, a=a, tau=tau)
        # cir is of format: cir[user_nr, AP_nr, time_index]
        

        print('Saving filtered_data')
        fn = 'filtered_data.pkl'
        with open(fn, "wb") as f:
            pickle.dump((cir, rxPos_grid, txPos, a, tau, time_axis), f)
        
        csv_filename = "sensorpositions.csv"
        # Write to CSV
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["mask", "x", "y", "z"])  # header

            for mask, (x, y, z) in zip(save_mask, rxPos_grid):
                writer.writerow([mask, x, y, z])
                
        csv_filename = "APpositions.csv"
        # Write to CSV
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["mask", "x", "y", "z"])  # header

            for x, y, z in txPos:
                writer.writerow([1, x, y, z]) 
        
    
    

