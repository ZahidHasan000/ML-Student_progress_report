// import React, { useEffect, useState } from 'react';

// const PlotDisplay = () => {
//     const [plotFilenames, setPlotFilenames] = useState([]);

//     useEffect(() => {
//         // Function to fetch plot filenames from Flask API
//         const fetchPlots = async () => {
//             try {
//                 const response = await fetch('/api/analysis');
//                 const data = await response.json();
//                 if (data.plot_filenames && data.plot_filenames.length > 0) {
//                     setPlotFilenames(data.plot_filenames);
//                 } else {
//                     console.log('No plots available.');
//                 }
//             } catch (error) {
//                 console.error('Error fetching plots:', error);
//             }
//         };

//         // Call the fetchPlots function when the component mounts
//         fetchPlots();
//     }, []);

//     return (
//         <div>
//             <h1>Plots</h1>
//             {plotFilenames.length > 0 ? (
//                 <div>
//                     {plotFilenames.map((filename, index) => (
//                         <img key={index} src={filename} alt={`Plot ${index}`} style={{ width: '400px', height: 'auto', margin: '10px' }} />
//                     ))}
//                 </div>
//             ) : (
//                 <p>No plots available.</p>
//             )}
//         </div>
//     );
// };

// export default PlotDisplay;



import React, { useState } from 'react';
import axios from 'axios';

const DataAnalysis = () => {
    const [message, setMessage] = useState('');
    const [plotFilenames, setPlotFilenames] = useState([]);

    const fetchData = () => {
        axios.get('http://127.0.0.1:5000/api/analysis')
            .then((response) => {
                const { message, plot_filenames: plotFilenames } = response.data;
                setMessage(message);
                setPlotFilenames(plotFilenames);
            })
            .catch((error) => {
                console.error('Error fetching data:', error);
            });
    };

    return (
        <div>
            <button onClick={fetchData}>Fetch Data</button>
            <p>{message}</p>
            <div>
                {plotFilenames.map((filename, index) => (
                    <img key={index} src={filename} alt={`Plot ${index + 1}`} />
                ))}
            </div>
        </div>
    );
};

export default DataAnalysis;
