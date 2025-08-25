import React, { useState, useRef } from "react";
import Sidebar from "./components/Sidebar";
import CameraView from "./components/CameraView";
import './App.css';

function App() {
  const [cameraOn, setCameraOn] = useState(false);

  return (
    <div style={{
            display: 'flex',
            height: '100vh',
            textAlign: 'center',
            width: '100vw'
        }}>

       

    <div style={{ flex: 1 }}>
      {/* Lado izquierdo */}
      <Sidebar 
        cameraOn={cameraOn}
        onStartCamera={() => setCameraOn(true)} 
        onStopCamera={() => setCameraOn(false)}
        />

    </div>
      {/* Lado derecho */}
      <div style={{ flex: 1 }}>
        {cameraOn ? (
          <CameraView />
        ) : (
          <p className="text-gray-600">La cámara se mostrará aquí</p>
        )}
      </div>
    

     </div>
  );
}

export default App;
