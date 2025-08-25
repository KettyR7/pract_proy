import React from "react";
import logo from "../assets/logo.png";

function Sidebar({ cameraOn, onStartCamera, onStopCamera }) {
  return (
    <div className="w-1/3 bg-white shadow-md flex flex-col items-center" >
      <p>
        <img src={logo} alt="App Logo" className="w-32 mb-6"  />
      </p>  
      <button
        onClick={onStartCamera}
        className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700"
      >
        Iniciar Cámara
      </button>
      <button
        onClick={onStopCamera}
        className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700"
      >
        Apagar Cámara
      </button>
    </div>
  );
}

export default Sidebar;
