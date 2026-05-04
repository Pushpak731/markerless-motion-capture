import React, { useState, useEffect, Suspense, useRef, useCallback } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Environment, ContactShadows, useGLTF } from '@react-three/drei'
import { X, Upload, Box, Activity, Play, FileText } from 'lucide-react'
import * as THREE from 'three'

const MEDIAPIPE_JOINT_NAMES = [
  'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
  'right_eye_inner', 'right_eye', 'right_eye_outer',
  'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
  'left_index', 'right_index', 'left_thumb', 'right_thumb',
  'left_hip', 'right_hip', 'left_knee', 'right_knee',
  'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
  'left_foot_index', 'right_foot_index',
]

const BONE_MAP = {
  mixamorigLeftArm: ['left_shoulder', 'left_elbow'],
  mixamorigLeftForeArm: ['left_elbow', 'left_wrist'],
  mixamorigRightArm: ['right_shoulder', 'right_elbow'],
  mixamorigRightForeArm: ['right_elbow', 'right_wrist'],
  mixamorigLeftUpLeg: ['left_hip', 'left_knee'],
  mixamorigLeftLeg: ['left_knee', 'left_ankle'],
  mixamorigRightUpLeg: ['right_hip', 'right_knee'],
  mixamorigRightLeg: ['right_knee', 'right_ankle'],
  mixamorigSpine: ['__mid_hip__', '__mid_sho__'],
  mixamorigNeck: ['__mid_sho__', 'nose'],
  Skeleton_arm_joint_L__4_: ['left_shoulder', 'left_elbow'],
  Skeleton_arm_joint_L__3_: ['left_elbow', 'left_wrist'],
  Skeleton_arm_joint_R: ['right_shoulder', 'right_elbow'],
  Skeleton_arm_joint_R__2_: ['right_elbow', 'right_wrist'],
  leg_joint_L_1: ['left_hip', 'left_knee'],
  leg_joint_L_2: ['left_knee', 'left_ankle'],
  leg_joint_R_1: ['right_hip', 'right_knee'],
  leg_joint_R_2: ['right_knee', 'right_ankle'],
  Skeleton_torso_joint_1: ['__mid_hip__', '__mid_sho__'],
}

const BONE_RETARGET_SPECS = {
  mixamorigLeftArm: { gain: 0.55, limit: 60, alpha: 0.16 },
  mixamorigLeftForeArm: { gain: 0.40, limit: 45, alpha: 0.12 },
  mixamorigRightArm: { gain: 0.55, limit: 60, alpha: 0.16 },
  mixamorigRightForeArm: { gain: 0.40, limit: 45, alpha: 0.12 },
  mixamorigLeftUpLeg: { gain: 0.22, limit: 35, alpha: 0.07, invert: true },
  mixamorigLeftLeg: { gain: 0.16, limit: 25, alpha: 0.05, invert: true },
  mixamorigRightUpLeg: { gain: 0.22, limit: 35, alpha: 0.07, invert: true },
  mixamorigRightLeg: { gain: 0.16, limit: 25, alpha: 0.05, invert: true },
  mixamorigSpine: { gain: 0.35, limit: 30, alpha: 0.10 },
  mixamorigNeck: { gain: 0.28, limit: 25, alpha: 0.10 },
  Skeleton_arm_joint_L__4_: { gain: 0.55, limit: 60, alpha: 0.16 },
  Skeleton_arm_joint_L__3_: { gain: 0.40, limit: 45, alpha: 0.12 },
  Skeleton_arm_joint_R: { gain: 0.55, limit: 60, alpha: 0.16 },
  Skeleton_arm_joint_R__2_: { gain: 0.40, limit: 45, alpha: 0.12 },
  leg_joint_L_1: { gain: 0.22, limit: 35, alpha: 0.07, invert: true },
  leg_joint_L_2: { gain: 0.16, limit: 25, alpha: 0.05, invert: true },
  leg_joint_R_1: { gain: 0.22, limit: 35, alpha: 0.07, invert: true },
  leg_joint_R_2: { gain: 0.16, limit: 25, alpha: 0.05, invert: true },
  Skeleton_torso_joint_1: { gain: 0.35, limit: 30, alpha: 0.10 },
}

const BONE_ALIASES = {
  mixamorigLeftArm: ['mixamorig:LeftArm', 'LeftArm', 'leftUpperArm', 'LeftShoulder'],
  mixamorigLeftForeArm: ['mixamorig:LeftForeArm', 'LeftForeArm', 'leftLowerArm', 'LeftElbow'],
  mixamorigRightArm: ['mixamorig:RightArm', 'RightArm', 'rightUpperArm', 'RightShoulder'],
  mixamorigRightForeArm: ['mixamorig:RightForeArm', 'RightForeArm', 'rightLowerArm', 'RightElbow'],
  mixamorigLeftUpLeg: ['mixamorig:LeftUpLeg', 'LeftUpLeg', 'leftUpperLeg', 'LeftHip'],
  mixamorigLeftLeg: ['mixamorig:LeftLeg', 'LeftLeg', 'leftLowerLeg', 'LeftKnee'],
  mixamorigRightUpLeg: ['mixamorig:RightUpLeg', 'RightUpLeg', 'rightUpperLeg', 'RightHip'],
  mixamorigRightLeg: ['mixamorig:RightLeg', 'RightLeg', 'rightLowerLeg', 'RightKnee'],
  mixamorigSpine: ['mixamorig:Spine', 'Spine', 'spine', 'Hips'],
  mixamorigNeck: ['mixamorig:Neck', 'Neck', 'neck'],
}

const emptyMappingStatus = {
  mapped: 0,
  total: Object.keys(BONE_MAP).length,
  names: [],
}

function normalizeName(name) {
  return String(name || '').toLowerCase().replace(/[^a-z0-9]/g, '')
}

function findBone(scene, boneName) {
  const candidates = [boneName, ...(BONE_ALIASES[boneName] || [])]
  const candidateKeys = candidates.map(normalizeName)
  let suffixMatch = null

  scene.traverse((child) => {
    if (!child.isBone || suffixMatch?.name === child.name) return
    const itemKey = normalizeName(child.name)

    if (candidateKeys.includes(itemKey)) {
      suffixMatch = child
      return
    }

    if (!suffixMatch && candidateKeys.some((key) => itemKey.endsWith(key))) {
      suffixMatch = child
    }
  })

  return suffixMatch
}

function getJoint(frame, name) {
  if (name === '__mid_hip__') return frame.midHip
  if (name === '__mid_sho__') return frame.midShoulder
  return frame.joints[name]
}

function makePoseVector(frame, startName, endName) {
  const start = getJoint(frame, startName)
  const end = getJoint(frame, endName)
  if (!start || !end) return null

  return new THREE.Vector3(
    end.x - start.x,
    -(end.z - start.z),
    -(end.y - start.y),
  )
}

function parseRaw3dNodesCsv(text) {
  const rows = text.trim().split(/\r?\n/)
  if (rows.length < 2) return []

  const headers = rows[0].split(',').map((header) => header.trim())
  const requiredHeaders = ['timestamp_ms', 'nose_x', 'left_hip_x', 'right_hip_x']
  if (!requiredHeaders.every((header) => headers.includes(header))) return []

  return rows.slice(1).map((line, rowIndex) => {
    const values = line.split(',')
    const row = Object.fromEntries(headers.map((header, index) => [header, values[index]]))
    const joints = {}

    for (const jointName of MEDIAPIPE_JOINT_NAMES) {
      joints[jointName] = {
        x: Number.parseFloat(row[`${jointName}_x`] || '0'),
        y: Number.parseFloat(row[`${jointName}_y`] || '0'),
        z: Number.parseFloat(row[`${jointName}_z`] || '0'),
        v: Number.parseFloat(row[`${jointName}_v`] || '0'),
      }
    }

    const leftHip = joints.left_hip
    const rightHip = joints.right_hip
    const leftShoulder = joints.left_shoulder
    const rightShoulder = joints.right_shoulder

    return {
      frameIdx: Number.parseInt(row.frame_idx || String(rowIndex), 10),
      timestampMs: Number.parseFloat(row.timestamp_ms || '0'),
      root: {
        x: Number.parseFloat(row.root_x || String((leftHip.x + rightHip.x) / 2)),
        y: Number.parseFloat(row.root_y || String((leftHip.y + rightHip.y) / 2)),
        z: Number.parseFloat(row.root_z || String((leftHip.z + rightHip.z) / 2)),
      },
      midHip: {
        x: (leftHip.x + rightHip.x) / 2,
        y: (leftHip.y + rightHip.y) / 2,
        z: (leftHip.z + rightHip.z) / 2,
      },
      midShoulder: {
        x: (leftShoulder.x + rightShoulder.x) / 2,
        y: (leftShoulder.y + rightShoulder.y) / 2,
        z: (leftShoulder.z + rightShoulder.z) / 2,
      },
      joints,
    }
  })
}

function AvatarModel({ url, isLive, poseFrame, onMappingStatus }) {
  const { scene } = useGLTF(url)
  const modelRef = useRef()
  const mappedBonesRef = useRef({})
  const rootOriginRef = useRef(null)
  const smoothedRootRef = useRef(new THREE.Vector3())

  useEffect(() => {
    const mappedBones = {}

    Object.entries(BONE_MAP).forEach(([boneName, segment]) => {
      const bone = findBone(scene, boneName)
      if (!bone) return

      let childBone = bone.children.find((child) => child.isBone)
      if (!childBone && bone.children.length > 0) childBone = bone.children[0]

      const avatarRestVec = childBone ? childBone.position.clone() : new THREE.Vector3(0, 1, 0)
      if (avatarRestVec.lengthSq() < 1e-8) avatarRestVec.set(0, 1, 0)
      avatarRestVec.normalize()

      mappedBones[boneName] = {
        bone,
        segment,
        initialLocalQuat: bone.quaternion.clone(),
        avatarRestVec,
      }
    })

    mappedBonesRef.current = mappedBones
    rootOriginRef.current = null
    smoothedRootRef.current.set(0, 0, 0)
    onMappingStatus({
      mapped: Object.keys(mappedBones).length,
      total: Object.keys(BONE_MAP).length,
      names: Object.values(mappedBones).map(({ bone }) => bone.name),
    })
  }, [scene, onMappingStatus])

  useFrame((state, delta) => {
    const mappedBones = mappedBonesRef.current

    if (poseFrame && Object.keys(mappedBones).length > 0) {
      const root = poseFrame.root || poseFrame.midHip
      const rawRoot = new THREE.Vector3(root.x * 4, -root.z * 4, -root.y * 4)

      if (!rootOriginRef.current) rootOriginRef.current = rawRoot.clone()
      const targetRoot = rawRoot.sub(rootOriginRef.current)
      smoothedRootRef.current.lerp(targetRoot, 0.25)

      if (modelRef.current) {
        modelRef.current.position.set(
          smoothedRootRef.current.x,
          -1 + smoothedRootRef.current.z,
          smoothedRootRef.current.y,
        )
      }

      Object.entries(mappedBones).forEach(([boneName, restInfo]) => {
        const [startName, endName] = restInfo.segment
        const vector = makePoseVector(poseFrame, startName, endName)
        if (!vector || vector.lengthSq() < 1e-8) return

        const startJoint = getJoint(poseFrame, startName)
        const endJoint = getJoint(poseFrame, endName)
        const confidence = startJoint?.v && endJoint?.v ? Math.min(startJoint.v, endJoint.v) : 1
        vector.normalize()

        const spec = BONE_RETARGET_SPECS[boneName] || { gain: 0.35, limit: 40, alpha: 0.1 }
        const axis = restInfo.avatarRestVec.clone().cross(vector)
        if (axis.lengthSq() < 1e-8) return

        const angle = THREE.MathUtils.radToDeg(restInfo.avatarRestVec.angleTo(vector))
        let finalAngle = THREE.MathUtils.clamp(angle * spec.gain, -spec.limit, spec.limit)
        if (spec.invert) finalAngle = -finalAngle

        axis.normalize()
        const deltaQuat = new THREE.Quaternion().setFromAxisAngle(axis, THREE.MathUtils.degToRad(finalAngle))
        const targetQuat = restInfo.initialLocalQuat.clone().premultiply(deltaQuat)
        const alpha = confidence > 0.5 ? spec.alpha : spec.alpha * 0.4
        restInfo.bone.quaternion.slerp(targetQuat, alpha)
      })

      return
    }

    if (!isLive) return

    const spine = scene.getObjectByName('mixamorigSpine') || scene.getObjectByName('Spine')
    if (spine) spine.rotation.x += Math.sin(state.clock.elapsedTime * 2) * delta * 0.03
  })

  return <primitive ref={modelRef} object={scene} scale={1} position={[0, -1, 0]} />
}

export default function AvatarStudio({ onClose }) {
  const [glbFile, setGlbFile] = useState(null)
  const [modelUrl, setModelUrl] = useState(null)
  const [dataSource, setDataSource] = useState('live')
  const [csvFile, setCsvFile] = useState(null)
  const [poseFrames, setPoseFrames] = useState([])
  const [csvError, setCsvError] = useState('')
  const [frameIndex, setFrameIndex] = useState(0)
  const [mappingStatus, setMappingStatus] = useState(emptyMappingStatus)

  const handleMappingStatus = useCallback((status) => {
    setMappingStatus(status)
  }, [])

  const handleModelFileChange = (file) => {
    if (modelUrl) URL.revokeObjectURL(modelUrl)
    setGlbFile(file)
    setMappingStatus(emptyMappingStatus)
    setModelUrl(file ? URL.createObjectURL(file) : null)
  }

  const handleCsvFileChange = (file) => {
    setCsvFile(file)
    setFrameIndex(0)

    if (!file) {
      setPoseFrames([])
      setCsvError('')
    }
  }

  useEffect(() => {
    return () => {
      if (modelUrl) URL.revokeObjectURL(modelUrl)
    }
  }, [modelUrl])

  useEffect(() => {
    if (!csvFile) return undefined

    let cancelled = false
    csvFile.text().then((text) => {
      if (cancelled) return

      const frames = parseRaw3dNodesCsv(text)
      if (frames.length === 0) {
        setCsvError('This does not look like a raw 3D nodes CSV. Pick a file ending in _raw_3d_nodes.csv.')
        setPoseFrames([])
        setFrameIndex(0)
        return
      }

      setCsvError('')
      setPoseFrames(frames)
      setFrameIndex(0)
    }).catch(() => {
      if (!cancelled) setCsvError('Could not read the CSV file.')
    })

    return () => {
      cancelled = true
    }
  }, [csvFile])

  useEffect(() => {
    if (dataSource !== 'offline' || poseFrames.length === 0) return undefined

    const interval = window.setInterval(() => {
      setFrameIndex((current) => (current + 1) % poseFrames.length)
    }, 33)

    return () => window.clearInterval(interval)
  }, [dataSource, poseFrames.length])

  const poseFrame = dataSource === 'offline' && poseFrames.length > 0
    ? poseFrames[frameIndex]
    : null

  return (
    <div className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-md flex items-center justify-center p-6">
      <div className="w-full max-w-6xl h-full max-h-[85vh] bg-espresso border border-mocha/30 rounded-2xl shadow-2xl flex flex-col overflow-hidden relative">
        <div className="h-16 border-b border-mocha/30 flex items-center justify-between px-6 bg-espresso/50">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-900/30 rounded-lg">
              <Box className="w-5 h-5 text-cyan-400" />
            </div>
            <h2 className="text-lg font-bold text-cream">3D Avatar Studio</h2>
            <span className="px-2 py-0.5 rounded text-xs font-mono bg-cyan-900/50 text-cyan-300 border border-cyan-500/20">
              BETA
            </span>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-mocha/20 rounded-lg text-mocha transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 flex flex-col lg:flex-row min-h-0">
          <div className="w-full lg:w-80 border-r border-mocha/30 p-6 flex flex-col gap-6 overflow-y-auto bg-latte/5">
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-latte">1. Select Character Model (.glb)</h3>
              <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-mocha/40 rounded-xl hover:bg-mocha/10 hover:border-mocha transition-colors cursor-pointer group">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <Upload className="w-8 h-8 mb-3 text-mocha group-hover:text-latte transition-colors" />
                  <p className="mb-2 text-sm text-mocha group-hover:text-cream"><span className="font-semibold">Click to upload</span></p>
                  <p className="text-xs text-mocha/60">Mixamo or Cesium GLB</p>
                </div>
                <input
                  type="file"
                  accept=".glb,.gltf"
                  className="hidden"
                  onChange={(event) => handleModelFileChange(event.target.files?.[0] || null)}
                />
              </label>
              {glbFile && <p className="text-xs text-green-400 font-mono break-all">Loaded: {glbFile.name}</p>}
            </div>

            <div className="space-y-3 pt-6 border-t border-mocha/20">
              <h3 className="text-sm font-medium text-latte">2. Select Data Source</h3>

              <div className="flex gap-2">
                <button
                  onClick={() => setDataSource('live')}
                  className={`flex-1 py-2 px-3 rounded-lg flex items-center justify-center gap-2 text-sm font-medium transition-colors ${dataSource === 'live' ? 'bg-cyan-900/40 border border-cyan-500/50 text-cyan-300' : 'bg-espresso border border-mocha/30 text-mocha hover:text-latte'}`}
                >
                  <Activity className="w-4 h-4" /> Live WebCam
                </button>
                <button
                  onClick={() => setDataSource('offline')}
                  className={`flex-1 py-2 px-3 rounded-lg flex items-center justify-center gap-2 text-sm font-medium transition-colors ${dataSource === 'offline' ? 'bg-purple-900/40 border border-purple-500/50 text-purple-300' : 'bg-espresso border border-mocha/30 text-mocha hover:text-latte'}`}
                >
                  <Play className="w-4 h-4" /> Offline CSV
                </button>
              </div>

              {dataSource === 'offline' && (
                <div className="mt-4 space-y-3">
                  <label className="flex flex-col items-center justify-center w-full h-24 border-2 border-dashed border-purple-500/30 rounded-xl hover:bg-purple-900/10 hover:border-purple-400/50 transition-colors cursor-pointer group">
                    <FileText className="w-6 h-6 mb-2 text-purple-300" />
                    <p className="text-sm text-purple-100"><span className="font-semibold">Load raw 3D nodes CSV</span></p>
                    <p className="text-xs text-purple-200/60">*_raw_3d_nodes.csv</p>
                    <input
                      type="file"
                      accept=".csv"
                      className="hidden"
                      onChange={(event) => handleCsvFileChange(event.target.files?.[0] || null)}
                    />
                  </label>
                  {csvFile && <p className="text-xs text-green-400 font-mono break-all">CSV: {csvFile.name}</p>}
                  {poseFrames.length > 0 && (
                    <p className="text-xs text-purple-100/80 font-mono">
                      Frames: {frameIndex + 1} / {poseFrames.length}
                    </p>
                  )}
                  {csvError && <p className="text-xs text-red-300">{csvError}</p>}
                </div>
              )}
            </div>

            {modelUrl && (
              <div className="space-y-2 pt-6 border-t border-mocha/20">
                <h3 className="text-sm font-medium text-latte">3. Rig Mapping</h3>
                <div className="p-4 rounded-xl border border-cyan-500/20 bg-cyan-900/10">
                  <p className="text-xs text-cyan-100 font-mono">
                    Bones mapped: {mappingStatus.mapped} / {mappingStatus.total}
                  </p>
                  {mappingStatus.mapped === 0 && (
                    <p className="mt-2 text-xs text-orange-200">
                      No supported rig bones were found. Use a Mixamo-style GLB or add its bone names to the mapping aliases.
                    </p>
                  )}
                  {mappingStatus.names.length > 0 && (
                    <p className="mt-2 text-[11px] leading-5 text-cyan-100/60 break-words">
                      {mappingStatus.names.slice(0, 8).join(', ')}
                      {mappingStatus.names.length > 8 ? '...' : ''}
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>

          <div className="flex-1 relative bg-gradient-to-b from-espresso to-black">
            {!modelUrl ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-mocha/50">
                <Box className="w-20 h-20 mb-4 opacity-20" />
                <p className="text-lg">Please load a .glb model to begin.</p>
              </div>
            ) : (
              <Canvas camera={{ position: [0, 1.5, 3], fov: 50 }}>
                <color attach="background" args={['#1a1515']} />
                <ambientLight intensity={0.5} />
                <directionalLight position={[5, 5, 5]} intensity={1} castShadow />
                <directionalLight position={[-5, 5, -5]} intensity={0.5} />

                <Suspense fallback={null}>
                  <AvatarModel
                    url={modelUrl}
                    isLive={dataSource === 'live'}
                    poseFrame={poseFrame}
                    onMappingStatus={handleMappingStatus}
                  />
                  <Environment preset="city" />
                  <ContactShadows position={[0, -1, 0]} opacity={0.5} scale={10} blur={2} far={4} />
                </Suspense>

                <OrbitControls target={[0, 1, 0]} />
              </Canvas>
            )}

            {modelUrl && (
              <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 bg-black/50 backdrop-blur rounded-lg border border-white/10">
                <div className={`w-2 h-2 rounded-full ${dataSource === 'live' ? 'bg-cyan-400 animate-pulse' : 'bg-purple-400'}`} />
                <span className="text-xs font-mono text-cream">
                  {dataSource === 'live'
                    ? 'WAITING FOR LIVE DATA...'
                    : poseFrames.length > 0
                      ? `PLAYING RAW CSV ${frameIndex + 1}/${poseFrames.length}`
                      : 'READY FOR RAW CSV DATA'}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
