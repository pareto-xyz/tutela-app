import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer
} from 'recharts';

export default function RevealTimeline({ plotData }) {
    const {start_date, end_date, counts } = plotData;


    return (
        <div className="col-md-12 col-lg-6">
            <div className="tornado-info col-12">
                <div className="panel-sub col-12">about your input</div>
                <div className="panel-title col-12">
                    TIMELINE OF REVEALS
                </div>
                <div className="panel-sub col-12">
                    This shows BLAH BLAH BLAH @WILL PLEASE FILL ME IN
                    <ResponsiveContainer width="100%" height={400}>
                        <BarChart data={plotData} >
                            <CartesianGrid />
                            <XAxis dataKey="start_date" />
                            <YAxis />
                            <Tooltip 
                                contentStyle={{ 'backgroundColor': '#404040', 'border': 'transparent', 'borderRadius': '5px' }} />
                            {/* <Legend /> */}
                            <Bar dataKey="deposit_address_reuse" stackId="a" fill="#8884d8" />
                            <Bar dataKey="unique_gas_price" stackId="a" fill="pink" />
                            <Bar dataKey="multi_denomination" stackId="a" fill="violet" />
                            <Bar dataKey="linked_transaction" stackId="a" fill="#82ca9d" />

                        </BarChart>
                    </ResponsiveContainer>

                </div>
            </div>
        </div>
    )
};