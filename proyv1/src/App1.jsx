
import './App.css';
import SectionOne from './components/SectionOne';
import SectionTwo from './components/SectionTwo';


function App1() {
  
  

  return (
    
        <div style={{
            display: 'flex',
            height: '100vh',
            width: '100vw'
        }}>
        <div style={{ flex: 1 }}>
            <SectionOne><div className="App"></div>
            </SectionOne>
        </div>
        <div style={{ flex: 1 }}>
            <SectionTwo />
        </div>
        </div>
     
    
  );
}

export default App1;