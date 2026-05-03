    def _apply_frame(self, frame_idx):
        if not self._frames or not self._controlled_joints:
            return

        frame = self._frames[frame_idx]
        joints = frame['joints']
        self._frame_idx = frame_idx

        # Update HUD
        status_text = "PLAYING" if self._playing else "PAUSED"
        self._hud_text.setText(
            f"Frame: {frame_idx + 1} / {len(self._frames)}\n"
            f"Time: {frame['timestamp_ms'] / 1000:.2f}s\n"
            f"Status: {status_text}"
        )
        
        # Helper for calculating vectors between joints
        def joint_vec(js, a_name, b_name):
            ja = js[a_name] if isinstance(a_name, str) else a_name
            jb = js[b_name] if isinstance(b_name, str) else b_name
            return Vec3(jb['x'] - ja['x'], -(jb['z'] - ja['z']), -(jb['y'] - ja['y']))

        # Calculate midpoints for stable torso/root
        l_hip, r_hip = joints['left_hip'], joints['right_hip']
        l_sho, r_sho = joints['left_shoulder'], joints['right_shoulder']
        
        mid_hip = {
            'x': (l_hip['x'] + r_hip['x']) / 2,
            'y': (l_hip['y'] + r_hip['y']) / 2,
            'z': (l_hip['z'] + r_hip['z']) / 2,
        }
        mid_sho = {
            'x': (l_sho['x'] + r_sho['x']) / 2,
            'y': (l_sho['y'] + r_sho['y']) / 2,
            'z': (l_sho['z'] + r_sho['z']) / 2,
        }

        # 1. Root Translation (with Centering)
        scale = 5.0
        raw_root = Vec3(mid_hip['x'] * scale, -mid_hip['z'] * scale, -mid_hip['y'] * scale)
        
        if not hasattr(self, "_root_origin"):
            self._root_origin = raw_root
            self._avatar.setZ(0)

        target_pos = raw_root - self._root_origin
        if not hasattr(self, "_last_root_pos"): self._last_root_pos = target_pos
        self._last_root_pos = self._last_root_pos * 0.8 + target_pos * 0.2
        self._avatar.setPos(self._last_root_pos)

        # 2. Capture Human Rest Pose (Frame 0)
        if frame_idx == 0:
            self._human_rest_joints = joints
            self._human_rest_mid_hip = mid_hip
            self._human_rest_mid_sho = mid_sho
            self._bone_rest_data = {} 

        # 3. Process Bones
        for bone_name, (start_name, end_name) in BONE_MAP.items():
            if DEBUG_ONLY_BONE and bone_name != DEBUG_ONLY_BONE:
                continue
                
            bone_np = self._controlled_joints.get(bone_name)
            if not bone_np: continue

            # Special Case: Torso uses midpoints
            if 'torso' in bone_name or 'Spine' in bone_name:
                h_rest_vec = joint_vec({'h': self._human_rest_mid_hip, 's': self._human_rest_mid_sho}, 'h', 's')
                h_now_vec = joint_vec({'h': mid_hip, 's': mid_sho}, 'h', 's')
                conf = 1.0
            else:
                h_rest_vec = joint_vec(self._human_rest_joints, start_name, end_name)
                h_now_vec = joint_vec(joints, start_name, end_name)
                conf = min(joints[start_name]['v'], joints[end_name]['v'])

            if h_rest_vec.length() < 1e-6 or h_now_vec.length() < 1e-6: continue
            
            h_rest_vec.normalize()
            h_now_vec.normalize()

            axis = h_rest_vec.cross(h_now_vec)
            angle = h_rest_vec.angleDeg(h_now_vec)
            
            limit = 120
            if 'torso' in bone_name: limit = 25
            elif 'leg' in bone_name: limit = 90
            
            damping = 0.8 if limit > 30 else 0.4
            final_angle = min(max(angle * damping, -limit), limit)

            h_delta_quat = LQuaternionf()
            if axis.length() > 1e-6:
                axis.normalize()
                h_delta_quat.setFromAxisAngle(final_angle, axis)

            if bone_name not in self._bone_rest_data:
                self._bone_rest_data[bone_name] = {'initial_local_quat': bone_np.getQuat()}
            
            rest_info = self._bone_rest_data[bone_name]
            target_q = h_delta_quat * rest_info['initial_local_quat']
            
            alpha = 0.15 if conf > 0.4 else 0.05
            current_q = bone_np.getQuat()
            smooth_q = current_q * (1.0 - alpha) + target_q * alpha
            smooth_q.normalize()
            bone_np.setQuat(smooth_q)

        self._avatar.update()
        if SHOW_DEBUG_SKELETON:
            self._draw_debug_skeleton(joints, mid_hip, mid_sho)

    def _draw_debug_skeleton(self, joints, mid_hip, mid_sho):
        from panda3d.core import LineSegs
        self._debug_np.node().removeAllChildren()
        ls = LineSegs()
        ls.setThickness(2.0)
        s = 2.0
        def p(j): return Vec3(j['x']*s, -j['z']*s, -j['y']*s)
        ls.setColor(1, 1, 0, 1)
        ls.moveTo(p(mid_hip))
        ls.drawTo(p(mid_sho))
        ls.setColor(0, 1, 1, 1)
        for _, (start, end) in BONE_MAP.items():
            if isinstance(start, str) and isinstance(end, str):
                ls.moveTo(p(joints[start]))
                ls.drawTo(p(joints[end]))
        self._debug_np.attachNewNode(ls.create())
