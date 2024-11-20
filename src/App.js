import logo from './logo.svg';
import './App.css';
import { UploadImage } from "./components/uploadImage";
import { Webcam } from "./components/webcam";


function App() {
  return (
    <div className="App">
      <header>
       <UploadImage />
          <div>
            <h1>Webcam</h1>
          </div>
          <Webcam />
      </header>
    </div>
  );
}

export default App;
