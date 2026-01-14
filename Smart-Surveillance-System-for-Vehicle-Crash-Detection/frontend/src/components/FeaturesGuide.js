import React, { useState } from 'react';
import './FeaturesGuide.css';

/**
 * FeaturesGuide Component
 * 
 * A user-friendly panel that explains all the AI capabilities of the
 * Smart Surveillance System. Helps users understand what the system
 * is detecting and how to interpret the results.
 */
const FeaturesGuide = () => {
    const [isExpanded, setIsExpanded] = useState(true);
    const [activeFeature, setActiveFeature] = useState(null);

    const features = [
        {
            id: 'crash',
            icon: '🚗',
            title: 'Crash Detection',
            shortDesc: 'AI-powered accident detection',
            longDesc: 'Uses YOLOv8 deep learning to detect vehicle crashes in real-time with over 95% accuracy. The system identifies collisions, rollovers, and multi-vehicle accidents.',
            color: '#ef4444'
        },
        {
            id: 'severity',
            icon: '⚠️',
            title: 'Severity Triage',
            shortDesc: 'Classifies crash severity levels',
            longDesc: 'Automatically classifies crashes into three categories: Severe (immediate emergency response), Moderate (medical attention needed), and Mild (minor incident). Uses velocity analysis and impact detection.',
            color: '#f59e0b'
        },
        {
            id: 'speed',
            icon: '🏎️',
            title: 'Speed Tracking',
            shortDesc: 'Estimates vehicle speeds',
            longDesc: 'Tracks each vehicle and estimates speed in km/h using trajectory analysis. Color-coded indicators show: Green (safe), Yellow (moderate), Red (high speed/dangerous).',
            color: '#22c55e'
        },
        {
            id: 'collision',
            icon: '⏱️',
            title: 'Collision Warning',
            shortDesc: 'Predicts potential crashes',
            longDesc: 'Calculates Time-to-Collision (TTC) between vehicles. Warns about potential collisions before they happen, enabling preventive measures. Shows risk levels and countdown timers.',
            color: '#3b82f6'
        },
        {
            id: 'behavior',
            icon: '🚨',
            title: 'Behavior Analysis',
            shortDesc: 'Detects dangerous driving',
            longDesc: 'Identifies dangerous behaviors: Swerving, Wrong-way driving, Sudden braking, Aggressive acceleration, Erratic lane changes, and Tailgating. Each alert includes severity level.',
            color: '#a855f7'
        },
        {
            id: 'privacy',
            icon: '👤',
            title: 'Privacy Protection',
            shortDesc: 'GDPR-compliant anonymization',
            longDesc: 'Automatically blurs faces and license plates in real-time using edge-based AI. Ensures privacy compliance while maintaining detection accuracy. All alerts are anonymized before transmission.',
            color: '#06b6d4'
        }
    ];

    return (
        <div className={`features-guide ${isExpanded ? 'features-guide--expanded' : 'features-guide--collapsed'}`}>
            <header className="features-guide__header" onClick={() => setIsExpanded(!isExpanded)}>
                <div className="features-guide__title-group">
                    <span className="features-guide__icon">📖</span>
                    <h3 className="features-guide__title">What This System Does</h3>
                </div>
                <button className="features-guide__toggle" aria-label="Toggle features guide">
                    <span className={`features-guide__arrow ${isExpanded ? 'features-guide__arrow--up' : ''}`}>▼</span>
                </button>
            </header>

            {isExpanded && (
                <div className="features-guide__content">
                    <p className="features-guide__intro">
                        Our AI-powered surveillance system provides real-time vehicle crash detection and emergency response.
                        Click any feature below to learn more.
                    </p>

                    <div className="features-guide__grid">
                        {features.map((feature) => (
                            <div
                                key={feature.id}
                                className={`feature-card ${activeFeature === feature.id ? 'feature-card--active' : ''}`}
                                style={{ '--feature-color': feature.color }}
                                onClick={() => setActiveFeature(activeFeature === feature.id ? null : feature.id)}
                                role="button"
                                tabIndex={0}
                                onKeyPress={(e) => e.key === 'Enter' && setActiveFeature(activeFeature === feature.id ? null : feature.id)}
                            >
                                <div className="feature-card__icon-wrapper">
                                    <span className="feature-card__icon">{feature.icon}</span>
                                </div>
                                <div className="feature-card__content">
                                    <h4 className="feature-card__title">{feature.title}</h4>
                                    <p className="feature-card__desc">{feature.shortDesc}</p>
                                </div>
                                <div className="feature-card__indicator"></div>
                            </div>
                        ))}
                    </div>

                    {activeFeature && (
                        <div className="features-guide__detail">
                            <div className="feature-detail">
                                <span className="feature-detail__icon">
                                    {features.find(f => f.id === activeFeature)?.icon}
                                </span>
                                <div className="feature-detail__content">
                                    <h4 className="feature-detail__title">
                                        {features.find(f => f.id === activeFeature)?.title}
                                    </h4>
                                    <p className="feature-detail__desc">
                                        {features.find(f => f.id === activeFeature)?.longDesc}
                                    </p>
                                </div>
                                <button
                                    className="feature-detail__close"
                                    onClick={() => setActiveFeature(null)}
                                    aria-label="Close detail"
                                >
                                    ✕
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default FeaturesGuide;
