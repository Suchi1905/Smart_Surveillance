import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './IncidentTimeline.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
        return (
            <div className="chart-tooltip glass">
                <p className="chart-tooltip__label">{payload[0].payload.time}</p>
                <p className="chart-tooltip__value">{payload[0].value} incidents</p>
            </div>
        );
    }
    return null;
};

const IncidentTimeline = () => {
    const [incidents, setIncidents] = useState([]);

    const processTimelineData = (crashes) => {
        const hourlyMap = new Map();
        crashes.forEach(crash => {
            const hour = new Date(crash.timestamp).getHours();
            hourlyMap.set(hour, (hourlyMap.get(hour) || 0) + 1);
        });
        const timeline = [];
        const now = new Date();
        for (let i = 23; i >= 0; i--) {
            const hour = (now.getHours() - i + 24) % 24;
            timeline.push({
                time: `${hour.toString().padStart(2, '0')}:00`,
                count: hourlyMap.get(hour) || 0
            });
        }
        return timeline;
    };

    useEffect(() => {
        fetch(`${API_URL}/api/v1/crashes`)
            .then(res => {
                if (!res.ok) throw new Error('Not available');
                return res.json();
            })
            .then(data => {
                if (data.crashes) {
                    setIncidents(processTimelineData(data.crashes));
                }
            })
            .catch(() => { });
    }, []);

    const mockTimeline = [
        { time: '00:00', count: 1 }, { time: '02:00', count: 0 },
        { time: '04:00', count: 2 }, { time: '06:00', count: 1 },
        { time: '08:00', count: 3 }, { time: '10:00', count: 2 },
        { time: '12:00', count: 5 }, { time: '14:00', count: 3 },
        { time: '16:00', count: 4 }, { time: '18:00', count: 6 },
        { time: '20:00', count: 2 }, { time: '22:00', count: 1 }
    ];

    const timelineData = incidents.length > 0 ? incidents : mockTimeline;
    const totalIncidents = timelineData.reduce((sum, item) => sum + item.count, 0);

    const firstHalf = timelineData.slice(0, Math.floor(timelineData.length / 2))
        .reduce((sum, item) => sum + item.count, 0);
    const secondHalf = timelineData.slice(Math.floor(timelineData.length / 2))
        .reduce((sum, item) => sum + item.count, 0);
    const trendUp = secondHalf > firstHalf;

    return (
        <div className="incident-timeline glass">
            <div className="incident-timeline__header">
                <div className="incident-timeline__title-row">
                    <span className="incident-timeline__icon">📈</span>
                    <h3 className="incident-timeline__title">24h Timeline</h3>
                </div>
                <div className="incident-timeline__trend">
                    <span className={`incident-timeline__trend-icon ${trendUp ? 'trend--up' : 'trend--down'}`}>
                        {trendUp ? '↗' : '↘'}
                    </span>
                    <span className="incident-timeline__trend-total">{totalIncidents} total</span>
                </div>
            </div>

            <div className="incident-timeline__chart">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={timelineData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis
                            dataKey="time"
                            stroke="#94a3b8"
                            tick={{ fill: '#94a3b8', fontSize: 10 }}
                            interval="preserveStartEnd"
                        />
                        <YAxis
                            stroke="#94a3b8"
                            tick={{ fill: '#94a3b8', fontSize: 10 }}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Line
                            type="monotone"
                            dataKey="count"
                            stroke="#22d3ee"
                            strokeWidth={2}
                            dot={{ fill: '#22d3ee', r: 3 }}
                            activeDot={{ r: 5, fill: '#22d3ee', strokeWidth: 2, stroke: '#0ea5e9' }}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default IncidentTimeline;
