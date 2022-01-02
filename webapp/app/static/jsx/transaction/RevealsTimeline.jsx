import React, { useEffect, useState } from 'react';
import {
    BarChart, Bar, XAxis, YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer
} from 'recharts';
import { Form } from 'react-bootstrap';
import { getApi } from '../../js/utils';
import exampleResponse from '../../data/plot';

export default function RevealTimeline({ addr }) {
    // const {start_date, end_date, counts } = plotData;
    const [plotData, setPlotData] = useState([]);
    const [plotWindow, setPlotWindow] = useState('1yr');

    useEffect(() => {
        onSelectWindow('1yr');
    }, [])

    function onSelectWindow(windowOption) {
        setPlotWindow(windowOption);
        getApi(`/plot/transaction?address=${addr}&window=${windowOption}`, response => {
            // response = exampleResponse;
            const { query, data, success } = response.data;
            if (success === 1) {
                setPlotData(data);
            }
        })
    }

    return (
        <div className="col-md-12 col-lg-6">
            <div className="tornado-info col-12">
                <div className="panel-sub col-12">about your input</div>
                <div className="panel-title   row ">
                <div className="col-6 title-text">
                    TIMELINE OF REVEALS

                </div>
                <div className="col-6 select-window">
                    <Form.Select value={plotWindow} id="window" className="select-window-button" onChange={e => onSelectWindow(e.target.value)}>
                        <option value='1yr'>
                            past year
                        </option>
                        <option value='6mth'>
                            past 6 months
                        </option>
                        <option value='5yr'>
                            past 5 years
                        </option>
                    </Form.Select>
                </div>
                </div>
                <div className="panel-sub col-12">
                    This shows BLAH BLAH BLAH @WILL PLEASE FILL ME IN
                    <ResponsiveContainer width="100%" height={400} >
                        <BarChart data={plotData} margin={{ top: 5, right: 0, left: -30, bottom: 30 }}>
                            <CartesianGrid />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip
                                contentStyle={{ 'backgroundColor': '#404040', 'border': 'transparent', 'borderRadius': '5px' }} />
                            <Bar dataKey="deposit_address_reuse" stackId="a" fill="#8884d8" />
                            <Bar dataKey="unique_gas_price" stackId="a" fill="pink" />
                            <Bar dataKey="multi_denomination" stackId="a" fill="violet" />
                            <Bar dataKey="linked_transaction" stackId="a" fill="#82ca9d" />
                            <Bar dataKey="address_match" stackId="a" fill="lightblue" />
                            <Bar dataKey="torn_mine" stackId="a" fill="white" />

                            <Legend verticalAlign="bottom" height={5} />

                        </BarChart>

                    </ResponsiveContainer>

                </div>
            </div>
        </div>
    )
};