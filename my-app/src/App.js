// import logo from './logo.svg';
// import PlotDisplay from './component/ml';
import DataAnalysis from './component/ml';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        {/* <PlotDisplay /> */}
        <DataAnalysis />
        {/* <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a> */}
      </header>
    </div>
  );
}

export default App;
