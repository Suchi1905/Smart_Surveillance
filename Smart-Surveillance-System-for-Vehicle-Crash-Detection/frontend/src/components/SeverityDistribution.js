import React, { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import './SeverityDistribution.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
        return (
            <div className="chart-tooltip glass">
                <p className="chart-tooltip__label">{payload[0].name}</p>
                <p className="chart-tooltip__value">{payload[0].value} incidents</p>
            </div>
        );
    }
    return null;
};

const SeverityDistribution = () => {
    const [crashStats, setCrashStats] = useState({});

    useEffect(() => {
        fetch(`${API_URL}/api/v1/crashes/stats/summary`)
            .then(res => {
                if (!res.ok) throw new Error('Not available');
                return res.json();
            })
            .then(data => setCrashStats(data))
            .catch(() => { });
    }, []);

    const severityData = crashStats.severity_distribution
        ? [
            { name: 'Severe', value: crashStats.severity_distribution.Severe || 0, color: 'var(--danger)' },
            { name: 'Moderate', value: crashStats.severity_distribution.Moderate || 0, color: 'var(--warning)' },
            { name: 'Mild', value: crashStats.severity_distribution.Mild || 0, color: 'var(--brand-cyan)' }
        ]
        : [
            { name: 'Severe', value: 3, color: 'var(--danger)' },
            { name: 'Moderate', value: 8, color: 'var(--warning)' },
            { name: 'Mild', value: 12, color: 'var(--brand-cyan)' }
        ];

    const total = severityData.reduce((sum, d) => sum + d.value, 0);

    return (
        <div className="severity-distribution glass">
            <div className="severity-distribution__header">
                <span className="severity-distribution__icon">📊</span>
                <h3 className="severity-distribution__title">Severity Distribution</h3>
            </div>

            <div className="severity-distribution__chart">
                <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                        <Pie
                            data={severityData}
                            cx="50%"
                            cy="50%"
                            innerRadius={45}
                            outerRadius={75}
                            paddingAngle={5}
                            dataKey="value"
                        >
                            {severityData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                        </Pie>
                        <Tooltip content={<CustomTooltip />} />
                        <Legend
                            verticalAlign="bottom"
                            height={36}
                            formatter={(value) => <span style={{ color: 'var(--text-2)', fontSize: '12px' }}>{value}</span>}
                        />
                    </PieChart>
                </ResponsiveContainer>
            </div>

            <div className="severity-distribution__total">
                <span className="severity-distribution__total-label">Total</span>
                <span className="severity-distribution__total-value">{total}</span>
            </div>
        </div>
    );
};

export default SeverityDistribution;
