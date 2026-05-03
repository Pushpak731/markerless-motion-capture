import React, { useState, useEffect, Suspense, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Environment, ContactShadows } from '@react-three/drei'
import { useGLTF } from '@react-three/drei'
import { X, Upload, Box, Activity, Play } from 'lucide-react'
import * as THREE from 'three'

// The actual 3D Model Component
function AvatarModel({ url, isLive }) {
  const { scene, nodes } = useGLTF(url)
  const modelRef = useRef()

  useEffect(() => {
    // Traverse the loaded model to map bones.
    // Assuming standard Mixamo rigging for now.
    scene.traverse((child) => {
      if (child.isBone) {
        // console.log("Found Bone:", child.name)
        // Here we will eventually map the bone to our custom IK retargeting dictionary
      }
    })
  }, [scene])

  // Real-Time Retargeting Loop
  useFrame((state, delta) => {
    if (!isLive) return
    
    // Placeholder for actual retargeting logic.
    // In the future, this will read from a global store or context containing the latest 
    // 3D world landmarks from the MediaPipe Python backend via WebSockets.
    
    // Example: Gentle procedural breathing animation to show it's alive
    if (modelRef.current) {
       const spine = scene.getObjectByName('mixamorigSpine') || scene.getObjectByName('Spine')
       if (spine) {
         spine.rotation.x = Math.sin(state.clock.elapsedTime * 2) * 0.05
       }
    }
  })

  return <primitive ref={modelRef} object={scene} scale={1} position={[0, -1, 0]} />
}

export default function AvatarStudio({ onClose }) {
  const [glbFile, setGlbFile] = useState(null)
  const [modelUrl, setModelUrl] = useState(null)
  const [dataSource, setDataSource] = useState('live') // 'live' or 'offline'
  
  // Clean up object URL on unmount
  useEffect(() => {
    if (glbFile) {
      const url = URL.createObjectURL(glbFile)
      setModelUrl(url)
      return () => URL.revokeObjectURL(url)
    }
  }, [glbFile])

  return (
    <div className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-md flex items-center justify-center p-6">
      <div className="w-full max-w-6xl h-full max-h-[85vh] bg-espresso border border-mocha/30 rounded-2xl shadow-2xl flex flex-col overflow-hidden relative">
        
        {/* Header */}
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

        {/* Content */}
        <div className="flex-1 flex flex-col lg:flex-row min-h-0">
          
          {/* Controls Sidebar */}
          <div className="w-full lg:w-80 border-r border-mocha/30 p-6 flex flex-col gap-6 overflow-y-auto bg-latte/5">
            
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-latte">1. Select Character Model (.glb)</h3>
              <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-mocha/40 rounded-xl hover:bg-mocha/10 hover:border-mocha transition-colors cursor-pointer group">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <Upload className="w-8 h-8 mb-3 text-mocha group-hover:text-latte transition-colors" />
                  <p className="mb-2 text-sm text-mocha group-hover:text-cream"><span className="font-semibold">Click to upload</span></p>
                  <p className="text-xs text-mocha/60">Mixamo or VRM GLB</p>
                </div>
                <input 
                  type="file" 
                  accept=".glb,.gltf" 
                  className="hidden" 
                  onChange={(e) => setGlbFile(e.target.files[0])}
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
                 <div className="mt-4 p-4 rounded-xl border border-orange-500/30 bg-orange-900/10">
                   <p className="text-xs text-orange-200">
                     Note: Standard metrics CSVs do not contain raw 3D vectors. An offline raw-export script is required to drive the animation.
                   </p>
                 </div>
              )}
            </div>

          </div>

          {/* 3D Viewport */}
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
                  <AvatarModel url={modelUrl} isLive={dataSource === 'live'} />
                  <Environment preset="city" />
                  <ContactShadows position={[0, -1, 0]} opacity={0.5} scale={10} blur={2} far={4} />
                </Suspense>
                
                <OrbitControls target={[0, 1, 0]} />
              </Canvas>
            )}
            
            {/* Overlay Status */}
            {modelUrl && (
              <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 bg-black/50 backdrop-blur rounded-lg border border-white/10">
                <div className={`w-2 h-2 rounded-full ${dataSource === 'live' ? 'bg-cyan-400 animate-pulse' : 'bg-purple-400'}`} />
                <span className="text-xs font-mono text-cream">
                  {dataSource === 'live' ? 'WAITING FOR LIVE DATA...' : 'READY FOR CSV DATA'}
                </span>
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  )
}
