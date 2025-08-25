import React, { useRef, useEffect } from "react";

function CameraView() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    const enableCamera = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
    };
    enableCamera();
    return () => {
      // Apagar cámara al desmontar el componente
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <video
      ref={videoRef}
      autoPlay
      playsInline
      className="w-full h-full object-cover"
    />
  );
}

export default CameraView;
