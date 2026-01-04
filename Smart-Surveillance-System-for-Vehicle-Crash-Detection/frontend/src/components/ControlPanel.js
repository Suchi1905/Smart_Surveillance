import React from 'react';
import './ControlPanel.css';

const ControlPanel = ({ 
  confidence, 
  onConfidenceChange, 
  onStartStream, 
  onStopStream, 
  isStreaming 
}) => {
  const handleSliderChange = (e) => {
    onConfidenceChange(parseFloat(e.target.value));
  };

  return (
    <div className="control-panel">
      <h2>Detection Controls</h2>
      <div className="control-content">
        <div className="confidence-control">
          <label htmlFor="confidence">
            Confidence Threshold: <span className="confidence-value">{confidence.toFixed(1)}</span>
          </label>
          <input
            type="range"
            id="confidence"
            min="0.1"
            max="1.0"
            step="0.1"
            value={confidence}
            onChange={handleSliderChange}
            className="slider"
          />
          <div className="slider-labels">
            <span>0.1</span>
            <span>0.5</span>
            <span>1.0</span>
          </div>
        </div>
        
        <div className="button-group">
          {!isStreaming ? (
            <button 
              className="btn btn-start"
              onClick={onStartStream}
            >
              ▶ Start Detection
            </button>
          ) : (
            <button 
              className="btn btn-stop"
              onClick={onStopStream}
            >
              ⏹ Stop Detection
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;


