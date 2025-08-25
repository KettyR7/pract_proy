import { useState } from 'react';

function SectionOne() {
 const [mensaje, setMensaje] = useState('¡Haz clic en el botón!');

  const manejarClick = () => {
    setMensaje('¡Gracias por hacer clic!');
    };

    return (
    <div style={{ backgroundColor: '#f7faf5ff', height: '100%', textAlign: 'center' }}>
      <h1>{mensaje}</h1>
        <button onClick={manejarClick}>Clic aquí</button>
    </div>
  );

}

export default SectionOne;
